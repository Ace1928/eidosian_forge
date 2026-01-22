import unittest
from Cython import Shadow
from Cython.Compiler import Options, CythonScope, PyrexTypes, Errors
class TestShadow(unittest.TestCase):

    def test_all_types_in_shadow(self):
        cython_scope = CythonScope.create_cython_scope(None)
        missing_types = []
        for key in cython_scope.entries.keys():
            if key.startswith('__') and key.endswith('__'):
                continue
            if key in ('PyTypeObject', 'PyObject_TypeCheck'):
                continue
            if not hasattr(Shadow, key):
                missing_types.append(key)
        self.assertEqual(missing_types, [])

    def test_int_types_in_shadow(self):
        missing_types = []
        for int_name in Shadow.int_types:
            for sign in ['', 'u', 's']:
                name = sign + int_name
                if sign and (int_name in ['Py_UNICODE', 'Py_UCS4', 'Py_ssize_t', 'ssize_t', 'ptrdiff_t', 'Py_hash_t'] or name == 'usize_t'):
                    self.assertNotIn(name, dir(Shadow))
                    self.assertNotIn('p_' + name, dir(Shadow))
                    continue
                if not hasattr(Shadow, name):
                    missing_types.append(name)
                for ptr in range(1, 4):
                    ptr_name = 'p' * ptr + '_' + name
                    if not hasattr(Shadow, ptr_name):
                        missing_types.append(ptr_name)
        self.assertEqual(missing_types, [])

    def test_most_types(self):
        cython_scope = CythonScope.create_cython_scope(None)

        class Context:
            cpp = False
            language_level = 3
            future_directives = []
        cython_scope.context = Context
        Errors.init_thread()
        missing_types = []
        missing_lookups = []
        for (signed, longness, name), type_ in PyrexTypes.modifiers_and_name_to_type.items():
            if name == 'object':
                continue
            if not hasattr(Shadow, name):
                missing_types.append(name)
            if not cython_scope.lookup_type(name):
                missing_lookups.append(name)
            for ptr in range(1, 4):
                ptr_name = 'p' * ptr + '_' + name
                if not hasattr(Shadow, ptr_name):
                    missing_types.append(ptr_name)
                if not cython_scope.lookup_type(ptr_name):
                    missing_lookups.append(ptr_name)
        self.assertEqual(missing_types, [])
        self.assertEqual(missing_lookups, [])