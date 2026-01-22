import pytest
from nipype.utils.functions import getsource, create_function_from_source
def _print_statement():
    try:
        exec('print("")')
        return True
    except SyntaxError:
        return False