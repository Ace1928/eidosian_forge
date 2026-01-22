from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import (

    Creates .unfiltered files from .sliced files.

    input: a .sliced file of logs (in ParlaiDialog format) from Mturk task 1, each of
        which starts with an initial prompt or topic request, and ends with a y_exp
    output: a .unfiltered file (in self-feeding format) with every utterance output by
        bot used as a label (i.e., act as though the bot was a human and we want to
        train in a normal supervised way).
    