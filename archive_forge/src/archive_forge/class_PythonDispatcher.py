import re
import torch._C as C
class PythonDispatcher:
    namespace = '__test__'
    name = 'foo'
    runtime_keys = ['CPU', 'AutogradCPU', 'FPGA', 'AutogradOther', 'XLA', 'AutogradXLA', 'Lazy', 'AutogradLazy']
    alias_keys = ['CompositeExplicitAutograd', 'Autograd', 'CompositeImplicitAutograd']
    supported_keys = runtime_keys + alias_keys

    def __init__(self):
        C._dispatch_check_invariants(self.name)
        self.ref = C._dispatch_library('FRAGMENT', self.namespace, '')
        self.ref.def_('foo(Tensor x) -> Tensor')
    '\n    Returns a list of dispatch keys supported by PythonDispatcher.\n    You can register kernels to these keys.\n    '

    def keys(self):
        return self.supported_keys
    "\n    Register kernels to the target dispatchKeys.\n    dispatchKeys(list[str]): a list of dispatch keys that you want to register\n      your own kernel. Note that you don't need to write the kernel yourself in\n      this PythonDispatcher.E.g. for CPU key, a kernel(e.g fn_CPU for CPU) is\n      automatically generated and registered.\n    "

    def register(self, dispatchKeys):
        if len(set(dispatchKeys)) != len(dispatchKeys):
            raise RuntimeError(f'Overriden is not allowed but found duplicates in {dispatchKeys}.')
        if 'CompositeImplicitAutograd' in dispatchKeys and 'CompositeExplicitAutograd' in dispatchKeys:
            raise RuntimeError('Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed.')
        for key in dispatchKeys:
            if key not in self.supported_keys:
                raise RuntimeError(f'{key} is not supported, please select a dispatch key in {self.supported_keys}.')
            self.ref.impl_t_t('foo', dispatch=key, debug='fn_' + key)
    '\n    Helper function to format (key, kernel).\n    '

    def _format_line(self, key, kernel):
        return f'{key:<15} {kernel}\n'
    '\n    Helper function to print a table header.\n    '

    def _format_header(self, header):
        s = f'\n{header}\n'
        s += self._format_line('key', 'kernel')
        s += '---------------------------\n'
        return s
    '\n    Returns raw output of all registration info for debugging only.\n    Use registrations() for a simplified version.\n    '

    def rawRegistrations(self):
        return C._dispatch_dump(f'{self.namespace}::{self.name}')
    '\n    Returns raw output of computed dispatch table for debugging only.\n    Use dispatchTable() for a simplified version.\n    '

    def rawDispatchTable(self):
        return C._dispatch_dump_table(f'{self.namespace}::{self.name}')
    '\n    Returns a table(str) including all the registrations from users.\n    Note this includes registrations to both runtime keys and alias keys.\n    '

    def registrations(self):
        output = self._format_header('Registered Kernels')
        state = self.rawRegistrations()
        state_entries = state.split('\n')
        for line in state_entries:
            first = line.split(':')[0]
            if any((first.startswith(k) for k in self.supported_keys)):
                kernel = line.split('::')[0].split(' ')[1]
                output += self._format_line(first, kernel)
        return output
    '\n    Returns the computed dispatch table(str). Note this only include\n    runtime keys, registrations to alias keys have been decoded to their\n    mapped runtime keys.\n    '

    def dispatchTable(self):
        output = self._format_header('Computed Dispatch Table')
        table = self.rawDispatchTable()
        table_entries = table.split('\n')
        regex = re.compile('registered at .*FallbackKernel\\.cpp.*(\\[)')
        for line in table_entries:
            k = line.split(':')[0]
            if k in self.runtime_keys:
                entry = regex.sub('[', line)
                output += self._format_line(k, entry.split(': ')[1])
        return output