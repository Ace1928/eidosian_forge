import pytest
import IPython.testing.tools as tt
def test_trust_help():
    pytest.importorskip('nbformat')
    tt.help_all_output_test('trust')