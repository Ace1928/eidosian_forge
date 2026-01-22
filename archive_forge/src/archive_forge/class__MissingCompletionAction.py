import argparse
import sys
class _MissingCompletionAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string):
        print('Install keyring[completion] for completion support.')
        parser.exit(0)