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
class StructuredRegisterDispatchKey(RegisterDispatchKey):
    g: NativeFunctionsGroup

    def gen_class_set_output_functions(self, k: SchemaKind, parent_class: str, generate_super: bool) -> str:
        if generate_super:
            set_output_super = f'{parent_class}::set_output_raw_strided(output_idx, sizes, strides, options, names);'
        else:
            set_output_super = ''

        def gen_set_output_function(name: str, maybe_create_proxy: bool) -> str:
            return f'\nvoid set_output_{name}(\n    int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,\n    TensorOptions options, DimnameList names\n) override {{\n{textwrap.indent(self.gen_class_set_output_body(k, maybe_create_proxy), '    ')}\n    if (!names.empty()) {{\n      namedinference::propagate_names(outputs_[output_idx], names);\n    }}\n    // super must happen after, so that downstream can use maybe_get_output\n    // to retrieve the output\n{textwrap.indent(set_output_super, '    ')}\n}}\n'
        return f'\n{gen_set_output_function('strided', maybe_create_proxy=True)}\n{gen_set_output_function('raw_strided', maybe_create_proxy=False)}\n'

    def gen_class_set_output_body(self, k: SchemaKind, maybe_create_proxy: bool) -> str:
        if self.backend_index.dispatch_key in [DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional]:
            maybe_set_guard = '\nauto current_device = guard_.current_device();\nif (C10_UNLIKELY(current_device.has_value())) {\n  TORCH_INTERNAL_ASSERT(*current_device == options.device(),\n    "structured kernels don\'t support multi-device outputs");\n} else {\n  guard_.reset_device(options.device());\n}\n'
            maybe_set_guard_line = maybe_set_guard + '\n'
        else:
            maybe_set_guard_line = maybe_set_guard = ''
        if maybe_create_proxy:
            create_proxy = '\nauto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);\nif (C10_UNLIKELY(maybe_proxy.has_value())) {\n    proxy_outputs_[output_idx] = std::move(maybe_proxy).value();\n}\n'
        else:
            create_proxy = ''
        if k is SchemaKind.functional:
            assert self.backend_index.dispatch_key in (DispatchKey.Meta, DispatchKey.CPU, DispatchKey.CUDA, DispatchKey.MPS, DispatchKey.CompositeExplicitAutogradNonFunctional)
            return f'{maybe_set_guard_line}\noutputs_[output_idx] = create_out(sizes, strides, options);'
        elif k is SchemaKind.inplace:
            return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\ncheck_inplace(out, sizes, options);\n{create_proxy}'
        elif k is SchemaKind.out:
            return f'{maybe_set_guard_line}\nconst auto& out = outputs_[output_idx].get();\nresize_out(out, sizes, strides, options);\n{create_proxy}'
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(f'{k} structured operators are currently not supported')
        else:
            assert_never(k)

    def gen_class_ctor(self, k: SchemaKind, class_name: str, returns: int) -> str:
        if k is SchemaKind.functional:
            return ''
        elif k is SchemaKind.inplace:
            return f'{class_name}(Tensor& self) : outputs_{{std::ref(self)}} {{}}'
        elif k is SchemaKind.out:
            out_args = ', '.join((f'Tensor& out{i}' for i in range(returns)))
            out_refs = ', '.join((f'std::ref(out{i})' for i in range(returns)))
            return f'{class_name}({out_args}) : outputs_{{ {out_refs} }} {{}}'
        elif k is SchemaKind.mutable or k is SchemaKind.scratch:
            raise AssertionError(f'{k} structured operators are currently not supported')
        else:
            assert_never(k)

    def gen_class(self, f: NativeFunction, k: SchemaKind, *, class_name: str, parent_class: str, generate_super: bool) -> str:
        if k is SchemaKind.functional:
            output_type = 'Tensor'
            output_value = 'outputs_[output_idx]'
            proxy_field = ''
        elif k is SchemaKind.inplace:
            output_type = 'std::reference_wrapper<Tensor>'
            output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
            proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
        elif k is SchemaKind.out:
            output_type = 'std::reference_wrapper<Tensor>'
            output_value = 'proxy_outputs_[output_idx].has_value() ? *proxy_outputs_[output_idx] : outputs_[output_idx].get()'
            proxy_field = f'std::array<c10::optional<Tensor>, {len(f.func.returns)}> proxy_outputs_;'
        if self.backend_index.dispatch_key == DispatchKey.CUDA:
            if self.rocm:
                guard_field = 'c10::hip::OptionalHIPGuardMasqueradingAsCUDA guard_;'
            else:
                guard_field = 'c10::cuda::OptionalCUDAGuard guard_;'
        elif self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
            guard_field = 'c10::OptionalDeviceGuard guard_;'
        elif self.backend_index.dispatch_key == DispatchKey.MPS:
            guard_field = 'c10::OptionalDeviceGuard guard_;'
        else:
            guard_field = ''
        indent = ' ' * 4
        class_ctor_str = self.gen_class_ctor(k, class_name, len(f.func.returns))
        lines = (f'struct {class_name} final : public {parent_class} {{', f'{textwrap.indent(class_ctor_str, indent)}', f'{textwrap.indent(self.gen_class_set_output_functions(k, parent_class, generate_super), indent)}', '    const Tensor& maybe_get_output(int64_t output_idx) override {', f'      return {output_value};\n', '    }', f'    std::array<{output_type}, {len(f.func.returns)}> outputs_;', f'{textwrap.indent(proxy_field, indent)}', f'{textwrap.indent(guard_field, indent)}', '};')
        return '\n'.join((line for line in lines if line))

    @method_with_native_function
    def gen_one(self, f: NativeFunction) -> Optional[str]:
        assert not f.manual_kernel_registration
        if self.target is Target.REGISTRATION and (not self.selector.is_native_function_selected(f)):
            return None
        if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional and f.func.kind() is SchemaKind.out:
            return None
        cpp_sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
        kern = self.backend_index.get_kernel(f)
        sig = NativeSignature(f.func, prefix=f'wrapper_{self.backend_index.dispatch_key}_', symint=kern is not None and kern.supports_symint())
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
            k = f.func.kind()
            sig_body = []
            context: List[Union[Binding, Expr]] = list(sig.arguments())
            if self.backend_index.dispatch_key is DispatchKey.Meta:
                class_name = f'structured_{meta.name(self.g)}_meta_{k.name}'
                parent_class = f'at::meta::structured_{meta.name(self.g)}'
            elif self.backend_index.dispatch_key is DispatchKey.CompositeExplicitAutogradNonFunctional:
                class_name = f'structured_{meta.name(self.g)}_default_backend_{k.name}'
                parent_class = f'at::meta::structured_{meta.name(self.g)}'
            else:
                metadata = self.backend_index.get_kernel(self.g)
                assert metadata is not None
                class_name = f'structured_{metadata.kernel}_{k.name}'
                parent_class = f'{metadata.cpp_namespace}::structured_{metadata.kernel}'
            if self.backend_index.device_guard:
                device_check_args = itertools.chain(f.func.arguments.out, f.func.arguments.flat_positional)
                sig_body.append(RegisterDispatchKey.gen_device_check(f.device_check, list(device_check_args), sig.name()))
            if k is SchemaKind.functional:
                sig_body.append(f'{class_name} op;')
            elif k is SchemaKind.inplace:
                sig_body.append(f'{class_name} op(self);')
            elif k is SchemaKind.out:
                out_args_str = ', '.join((a.name for a in f.func.arguments.out))
                sig_body.append(f'{class_name} op({out_args_str});')
            meta_exprs = ', '.join((e.expr for e in translate(context, structured.meta_arguments(self.g), method=False)))
            if self.g.out.precomputed:
                sig_body.append(f'auto precompute = op.meta({meta_exprs});')
                precomputed_values = [*self.g.out.precomputed.replace.values(), self.g.out.precomputed.add]
                for precomputed_elems in precomputed_values:
                    for arg in precomputed_elems:
                        context.append(Expr(expr=f'precompute.{arg.name}', type=structured.argument_type(arg, binds=arg.name)))
                sig_body.append('(void)precompute;')
            else:
                sig_body.append(f'op.meta({meta_exprs});')
            out_args = structured.out_arguments(self.g)
            for i, out_arg in enumerate(out_args):
                assert ConstRefCType(BaseCType(tensorT)) == out_arg.nctype.type
                if k is SchemaKind.out:
                    expr = f'op.maybe_get_output({i})'
                else:
                    expr = f'op.outputs_[{i}]'
                context.append(Expr(expr=expr, type=NamedCType(out_arg.nctype.name, MutRefCType(BaseCType(tensorT)))))
            if self.backend_index.dispatch_key == DispatchKey.CompositeExplicitAutogradNonFunctional:
                out_sig_group = CppSignatureGroup.from_native_function(self.g.out, method=False, fallback_binding=f.manual_cpp_binding)
                out_sig = out_sig_group.most_faithful_signature()
                api_name = out_sig.name()
                out_exprs = ', '.join((e.expr for e in translate(context, out_sig.arguments(), method=False)))
                sig_body.append(f'at::{api_name}({out_exprs});')
            elif self.backend_index.dispatch_key != DispatchKey.Meta:
                impl_exprs = ', '.join((e.expr for e in translate(context, structured.impl_arguments(self.g), method=False)))
                sig_body.append(f'op.impl({impl_exprs});')
            if k is SchemaKind.out or k is SchemaKind.inplace:
                for i in range(len(f.func.returns)):
                    sig_body.append(f'if (op.proxy_outputs_[{i}].has_value()) op.outputs_[{i}].get().copy_(*op.proxy_outputs_[{i}]);')
            if k is SchemaKind.functional:
                if len(f.func.returns) == 1:
                    ret_expr = 'std::move(op.outputs_[0])'
                else:
                    moved = ', '.join((f'std::move(op.outputs_[{i}])' for i in range(len(f.func.returns))))
                    ret_expr = f'std::make_tuple({moved})'
            elif k is SchemaKind.inplace:
                ret_expr = 'self'
            elif k is SchemaKind.out:
                if len(f.func.returns) == 1:
                    ret_expr = f.func.arguments.out[0].name
                else:
                    refs = ', '.join((a.name for a in f.func.arguments.out))
                    ret_expr = f'std::forward_as_tuple({refs})'
            sig_body.append(f'return {ret_expr};')
            sig_body_str = '\n'.join(sig_body)
            return f'{self.gen_class(f, k, class_name=class_name, parent_class=parent_class, generate_super=self.g.out.structured_inherits is not None)}\n\n{sig.defn()} {{\n{sig_body_str}\n}}\n'
        elif self.target is Target.REGISTRATION:
            return f'm.impl("{f.func.name}", TORCH_FN({sig.name()}));'
        else:
            assert_never(self.target)
            return None