import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def _render_uri_template(self, uri_template, params):
    encoded_params = {}
    for template_param in re.findall('{(.*?)}', uri_template):
        if template_param.endswith('+'):
            encoded_params[template_param] = percent_encode(params[template_param[:-1]], safe='/~')
        else:
            encoded_params[template_param] = percent_encode(params[template_param])
    return uri_template.format(**encoded_params)