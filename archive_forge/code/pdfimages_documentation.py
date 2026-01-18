import os
import reportlab
from reportlab import rl_config
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase import pdfdoc
from reportlab.lib.utils import isStr
from reportlab.lib.rl_accel import fp_str, asciiBase85Encode
from reportlab.lib.boxstuff import aspectRatioFix
Allow it to be used within pdfdoc framework.  This only
        defines how it is stored, not how it is drawn later.