import sys
import os.path
from lxml import etree as _etree # due to validator __init__ signature
@property
def validation_report(self):
    """ISO-schematron validation result report (None if result-storing has
        been turned off).
        """
    return self._validation_report