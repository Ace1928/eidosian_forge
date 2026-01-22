from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import enum
from googlecloudsdk.command_lib.survey import question
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import pkg_resources
def _LoadSurveyContent(self):
    """Loads the survey yaml file and return the parsed data."""
    survey_file = os.path.join(_GetSurveyContentDirectory(), self.name + '.yaml')
    survey_data = pkg_resources.GetResourceFromFile(survey_file)
    return yaml.load(survey_data)