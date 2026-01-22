import contextlib
import re
import tempfile
import numpy
import srsly
from thinc.api import get_current_ops
from spacy.tokens import Doc
from spacy.training import split_bilu_label
from spacy.util import make_tempdir  # noqa: F401
from spacy.vocab import Vocab
def apply_transition_sequence(parser, doc, sequence):
    """Perform a series of pre-specified transitions, to put the parser in a
    desired state."""
    for action_name in sequence:
        if '-' in action_name:
            move, label = split_bilu_label(action_name)
            parser.add_label(label)
    with parser.step_through(doc) as stepwise:
        for transition in sequence:
            stepwise.transition(transition)