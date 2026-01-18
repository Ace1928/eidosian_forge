from unittest import mock
import pytest
from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
from cirq.circuits.text_diagram_drawer import (
import cirq.testing as ct
def test_pick_charset():
    assert pick_charset(use_unicode=False, emphasize=False, doubled=False) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=False, doubled=True) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=True, doubled=False) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=True, doubled=True) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=False, doubled=False) == NORMAL_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=False, doubled=True) == DOUBLED_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=True, doubled=False) == BOLD_BOX_CHARS
    with pytest.raises(ValueError):
        pick_charset(use_unicode=True, emphasize=True, doubled=True)