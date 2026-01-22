import re
import csv
from typing import TYPE_CHECKING
from typing import Optional, List, Dict, Any, Iterator, Union, Tuple
def _get_data_object_for_decoding(matrix_type):
    if matrix_type == DENSE:
        return Data()
    elif matrix_type == COO:
        return COOData()
    elif matrix_type == LOD:
        return LODData()
    elif matrix_type == DENSE_GEN:
        return DenseGeneratorData()
    elif matrix_type == LOD_GEN:
        return LODGeneratorData()
    else:
        raise ValueError('Matrix type %s not supported.' % str(matrix_type))