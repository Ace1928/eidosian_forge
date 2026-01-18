import os
from pytest import raises
from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available
@require_bnb
def test_problematic_imports():
    with raises(RuntimeError, match='Please keep these imports'):
        import bitsandbytes as bnb
        notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)