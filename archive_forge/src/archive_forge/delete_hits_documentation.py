import argparse
import sys
import parlai.mturk.core.mturk_utils as mturk_utils

    This script should be used after some error occurs that leaves HITs live while the
    ParlAI MTurk server down.

    This will search through live HITs and list them by task ID, letting you close down
    HITs that do not link to any server and are thus irrecoverable.
    