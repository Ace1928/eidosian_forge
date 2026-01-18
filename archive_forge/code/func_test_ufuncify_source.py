import os
import tempfile
import shutil
from io import StringIO
from sympy.core import symbols, Eq
from sympy.utilities.autowrap import (autowrap, binary_function,
from sympy.utilities.codegen import (
from sympy.testing.pytest import raises
from sympy.testing.tmpfiles import TmpFileManager
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np
def test_ufuncify_source():
    x, y, z = symbols('x,y,z')
    code_wrapper = UfuncifyCodeWrapper(C99CodeGen('ufuncify'))
    routine = make_routine('test', x + y + z)
    source = get_string(code_wrapper.dump_c, [routine])
    expected = '#include "Python.h"\n#include "math.h"\n#include "numpy/ndarraytypes.h"\n#include "numpy/ufuncobject.h"\n#include "numpy/halffloat.h"\n#include "file.h"\n\nstatic PyMethodDef wrapper_module_%(num)sMethods[] = {\n        {NULL, NULL, 0, NULL}\n};\n\nstatic void test_ufunc(char **args, npy_intp *dimensions, npy_intp* steps, void* data)\n{\n    npy_intp i;\n    npy_intp n = dimensions[0];\n    char *in0 = args[0];\n    char *in1 = args[1];\n    char *in2 = args[2];\n    char *out0 = args[3];\n    npy_intp in0_step = steps[0];\n    npy_intp in1_step = steps[1];\n    npy_intp in2_step = steps[2];\n    npy_intp out0_step = steps[3];\n    for (i = 0; i < n; i++) {\n        *((double *)out0) = test(*(double *)in0, *(double *)in1, *(double *)in2);\n        in0 += in0_step;\n        in1 += in1_step;\n        in2 += in2_step;\n        out0 += out0_step;\n    }\n}\nPyUFuncGenericFunction test_funcs[1] = {&test_ufunc};\nstatic char test_types[4] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};\nstatic void *test_data[1] = {NULL};\n\n#if PY_VERSION_HEX >= 0x03000000\nstatic struct PyModuleDef moduledef = {\n    PyModuleDef_HEAD_INIT,\n    "wrapper_module_%(num)s",\n    NULL,\n    -1,\n    wrapper_module_%(num)sMethods,\n    NULL,\n    NULL,\n    NULL,\n    NULL\n};\n\nPyMODINIT_FUNC PyInit_wrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = PyModule_Create(&moduledef);\n    if (!m) {\n        return NULL;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "test", ufunc0);\n    Py_DECREF(ufunc0);\n    return m;\n}\n#else\nPyMODINIT_FUNC initwrapper_module_%(num)s(void)\n{\n    PyObject *m, *d;\n    PyObject *ufunc0;\n    m = Py_InitModule("wrapper_module_%(num)s", wrapper_module_%(num)sMethods);\n    if (m == NULL) {\n        return;\n    }\n    import_array();\n    import_umath();\n    d = PyModule_GetDict(m);\n    ufunc0 = PyUFunc_FromFuncAndData(test_funcs, test_data, test_types, 1, 3, 1,\n            PyUFunc_None, "wrapper_module_%(num)s", "Created in SymPy with Ufuncify", 0);\n    PyDict_SetItemString(d, "test", ufunc0);\n    Py_DECREF(ufunc0);\n}\n#endif' % {'num': CodeWrapper._module_counter}
    assert source == expected