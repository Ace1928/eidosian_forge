import logging
import sys
import textwrap
from absl import app
from absl import flags
import httplib2
import termcolor
import bigquery_client
import bq_auth_flags
import bq_flags
import bq_utils
import credential_loader
from auth import main_credential_loader
from frontend import utils as bq_frontend_utils
from utils import bq_logging
def _GetWaitPrinterFactoryFromFlags():
    """Returns the default wait_printer_factory to use while waiting for jobs."""
    if FLAGS.quiet:
        return BigqueryClient.QuietWaitPrinter
    if FLAGS.headless:
        return BigqueryClient.TransitionWaitPrinter
    return BigqueryClient.VerboseWaitPrinter