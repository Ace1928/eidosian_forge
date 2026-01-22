import itertools
import textwrap
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import torchgen.api.cpp as cpp
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function, native_function_manager
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import assert_never, mapMaybe, Target
@dataclass(frozen=True)
class RegisterDispatchKey:
    backend_index: BackendIndex
    target: Literal[Target.ANONYMOUS_DEFINITION, Target.NAMESPACED_DEFINITION, Target.NAMESPACED_DECLARATION, Target.REGISTRATION]
    selector: SelectiveBuilder
    rocm: bool
    symint: bool
    class_method_name: Optional[str]
    skip_dispatcher_op_registration: bool

    @staticmethod
    def gen_device_check(type: DeviceCheckType, args: List[Argument], method_name: str) -> str:
        if type == DeviceCheckType.NoCheck:
            return '  // No device check\n'
        device_check = 'c10::optional<Device> common_device = nullopt;\n'
        device_check += '(void)common_device; // Suppress unused variable warning\n'
        for arg in args:
            if arg.type.is_tensor_like():
                device_check += f'\n  c10::impl::check_and_update_common_device(common_device, {arg.name}, "{method_name}", "{arg.name}");'
        return device_check

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        if isinstance(f, NativeFunctionsGroup):
            g: NativeFunctionsGroup = f
            if g.structured:
                return self.gen_structured(g)
            else:
                return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        elif isinstance(f, NativeFunction):
            r = self.gen_unstructured(f)
            return [] if r is None else [r]
        else:
            assert_never(f)

    def wrapper_kernel_sig(self, f: NativeFunction) -> Union[NativeSignature, DispatcherSignature]:
        return DispatcherSignature.from_schema(f.func, prefix=f'wrapper_{self.backend_index.dispatch_key}_{f.func.name.overload_name}_', symint=self.symint)

    def gen_out_inplace_wrapper(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]) -> Optional[str]:
        if g is None:
            return None
        k = f.func.kind()
        if k is SchemaKind.inplace:
            copy_op = 'at::_copy_from'
        elif k is SchemaKind.out:
            copy_op = 'at::_copy_from_and_resize'
        else:
            raise AssertionError('gen_out_inplace_wrapper called on a functional op')
        sig = self.wrapper_kernel_sig(f)
        name = sig.name()
        func_res = f'{name}_tmp'
        return_names = cpp.return_names(f)
        if len(return_names) > 1:
            updates = '\n  '.join((f'{copy_op}(std::get<{i}>({func_res}), {ret_name});' for i, ret_name in enumerate(return_names)))
            returns = f'{sig.returns_type().cpp_type()}({', '.join(return_names)})'
        elif len(return_names) == 1:
            ret_name = return_names[0]
            updates = f'{copy_op}({func_res}, {ret_name});'
            returns = ret_name
        else:
            assert len(f.func.arguments.out) == 1
            returns = ''
            out_arg = f.func.arguments.out[0]
            if out_arg.type.is_list_like():
                updates = f'    for (int64_t i = 0; i < {func_res}.size(); ++i) {{\n        {copy_op}({func_res}[i], {out_arg.name}[i]);\n    }}'
            else:
                updates = f'{copy_op}({func_res}, {out_arg.name});'
        functional_sig = self.wrapper_kernel_sig(g.functional)
        wrapper_name = sig.name()
        return f'{sig.defn(name=wrapper_name)} {{\n  auto {func_res} = {functional_sig.name()}({', '.join((e.expr for e in translate(sig.arguments(), functional_sig.arguments())))});\n  {updates}\n  return {returns};\n}}\n'

    def gen_structured(self, g: NativeFunctionsGroup) -> List[str]:
        metadata = self.backend_index.get_kernel(g)
        if self.backend_index.dispatch_key == DispatchKey.Meta:
            assert not self.backend_index.has_kernel(g.out), 'Do not explicitly specify Meta dispatch key on structured functions, they will be automatically generated for you'
        elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
            assert not self.backend_index.has_kernel(g.out), 'Do not explicitly specify CompositeExplicitAutograd dispatch key on structured functions, they will be automatically generated for you'
        elif metadata is None or not metadata.structured:
            return list(mapMaybe(lambda f: self.gen_unstructured(f, g), g.functions()))
        structured_gen = StructuredRegisterDispatchKey(self.backend_index, self.target, self.selector, self.rocm, self.symint, self.class_method_name, self.skip_dispatcher_op_registration, g)
        return list(mapMaybe(structured_gen.gen_one, g.functions()))

    def gen_unstructured(self, f: NativeFunction, g: Optional[NativeFunctionsGroup]=None) -> Optional[str]:
        with native_function_manager(f):
            inplace_meta = False
            gets_out_inplace_wrapper = False
            if not self.backend_index.has_kernel(f):
                if self.backend_index.dispatch_key == DispatchKey.Meta and f.func.kind() is SchemaKind.inplace and (not f.has_composite_kernel) and (len(f.func.returns) == 1):
                    inplace_meta = True
                elif not self.backend_index.use_out_as_primary and g is not None and gets_generated_out_inplace_wrapper(f, g, self.backend_index):
                    gets_out_inplace_wrapper = True
                else:
                    return None
            if f.manual_kernel_registration:
                return None
            if self.target is Target.REGISTRATION and (not self.selector.is_native_function_selected(f)):
                return None
            sig = self.wrapper_kernel_sig(f)
            name = sig.name()
            returns_type = sig.returns_type().cpp_type()
            args = sig.arguments()
            args_str = ', '.join((a.defn() for a in args))
            cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
            if self.target is Target.NAMESPACED_DECLARATION:
                result = ''
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += f'TORCH_API {cpp_sig.decl()};\n'
                return result
            elif self.target is Target.NAMESPACED_DEFINITION:

                def generate_defn(cpp_sig: CppSignature) -> str:
                    return f'\n{cpp_sig.defn()} {{\nreturn {sig.name()}({', '.join((e.expr for e in translate(cpp_sig.arguments(), sig.arguments())))});\n}}\n'
                result = ''
                for cpp_sig in cpp_sig_group.signatures(symint=self.symint):
                    result += generate_defn(cpp_sig)
                return result
            elif self.target is Target.ANONYMOUS_DEFINITION:
                if inplace_meta:
                    assert f.func.arguments.self_arg is not None
                    self_arg_name = f.func.arguments.self_arg.argument.name
                    return f'\n{returns_type} {name}({args_str}) {{\n  TORCH_CHECK_NOT_IMPLEMENTED({self_arg_name}.is_meta(),\n    "Cannot inplace into non-meta tensor with meta tensor argument");\n  return {self_arg_name};\n}}\n'
                if gets_out_inplace_wrapper:
                    return self.gen_out_inplace_wrapper(f, g)
                metadata = self.backend_index.get_kernel(f)
                if metadata is None:
                    return None
                if self.class_method_name is None:
                    impl_name = f'{metadata.cpp_namespace}::{metadata.kernel}'
                else:
                    impl_name = f'{metadata.cpp_namespace}::{self.class_method_name}::{metadata.kernel}'
                kernel_sig = kernel_signature(f, self.backend_index)
                args_exprs_str = ', '.join((e.expr for e in translate(sig.arguments(), kernel_sig.arguments(), method=False)))
                device_check = '  // No device check\n'
                if self.backend_index.device_guard:
                    device_check_args = itertools.chain(f.func.arguments.out, f.func.arguments.flat_positional)
                    device_check = RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), name)
                device_guard = '// DeviceGuard omitted'
                if f.device_guard and self.backend_index.device_guard:
                    has_tensor_options = any((isinstance(a, TensorOptionsArguments) for a in f.func.arguments.non_out))
                    if has_tensor_options:
                        device_guard = '\n  const DeviceGuard device_guard(device_or_default(device));'
                        if is_cuda_dispatch_key(self.backend_index.dispatch_key):
                            device_guard = f'globalContext().lazyInitCUDA();\n{device_guard}'
                    else:
                        self_arg = [f.func.arguments.self_arg.argument] if f.func.arguments.self_arg is not None else []
                        candidate_args = itertools.chain(self_arg, f.func.arguments.out, f.func.arguments.flat_positional)
                        device_of = next((f'{a.name}' for a in candidate_args if a.type.is_tensor_like()), None)
                        if device_of is not None:
                            device_guard = f'const OptionalDeviceGuard device_guard(device_of({device_of}));'
                return f'namespace {{\n\n{returns_type} {name}({args_str}) {{\n  {device_check}\n\n  {device_guard}\n  return {impl_name}({args_exprs_str});\n}}\n\n}} // anonymous namespace\n'
            elif self.target is Target.REGISTRATION:
                if f.manual_kernel_registration or self.skip_dispatcher_op_registration:
                    return None
                else:
                    payload = f'TORCH_FN({name})'
                    return f'm.impl("{f.func.name}",\n{payload});\n'
            else:
                assert_never(self.target)