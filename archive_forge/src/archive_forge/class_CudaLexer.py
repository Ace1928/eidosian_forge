import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class CudaLexer(CLexer):
    """
    For NVIDIA `CUDA™ <http://developer.nvidia.com/category/zone/cuda-zone>`_
    source.

    .. versionadded:: 1.6
    """
    name = 'CUDA'
    filenames = ['*.cu', '*.cuh']
    aliases = ['cuda', 'cu']
    mimetypes = ['text/x-cuda']
    function_qualifiers = set(('__device__', '__global__', '__host__', '__noinline__', '__forceinline__'))
    variable_qualifiers = set(('__device__', '__constant__', '__shared__', '__restrict__'))
    vector_types = set(('char1', 'uchar1', 'char2', 'uchar2', 'char3', 'uchar3', 'char4', 'uchar4', 'short1', 'ushort1', 'short2', 'ushort2', 'short3', 'ushort3', 'short4', 'ushort4', 'int1', 'uint1', 'int2', 'uint2', 'int3', 'uint3', 'int4', 'uint4', 'long1', 'ulong1', 'long2', 'ulong2', 'long3', 'ulong3', 'long4', 'ulong4', 'longlong1', 'ulonglong1', 'longlong2', 'ulonglong2', 'float1', 'float2', 'float3', 'float4', 'double1', 'double2', 'dim3'))
    variables = set(('gridDim', 'blockIdx', 'blockDim', 'threadIdx', 'warpSize'))
    functions = set(('__threadfence_block', '__threadfence', '__threadfence_system', '__syncthreads', '__syncthreads_count', '__syncthreads_and', '__syncthreads_or'))
    execution_confs = set(('<<<', '>>>'))

    def get_tokens_unprocessed(self, text):
        for index, token, value in CLexer.get_tokens_unprocessed(self, text):
            if token is Name:
                if value in self.variable_qualifiers:
                    token = Keyword.Type
                elif value in self.vector_types:
                    token = Keyword.Type
                elif value in self.variables:
                    token = Name.Builtin
                elif value in self.execution_confs:
                    token = Keyword.Pseudo
                elif value in self.function_qualifiers:
                    token = Keyword.Reserved
                elif value in self.functions:
                    token = Name.Function
            yield (index, token, value)