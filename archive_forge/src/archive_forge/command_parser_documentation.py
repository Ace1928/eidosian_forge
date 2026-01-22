import argparse
import ast
import re
import sys
Get an ArgumentParser for a command that prints tensor values.

  Examples of such commands include print_tensor and print_feed.

  Args:
    description: Description of the ArgumentParser.

  Returns:
    An instance of argparse.ArgumentParser.
  