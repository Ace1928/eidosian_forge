from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def AddContainerImageField(parser, use_default=True):
    """Adds the --container-predefined-image and --container-custom-image flags to the given parser.
  """
    predefined_image_help_text = '  Code editor on base images.'
    custom_image_help_text = '  A docker image for the workstation. This image must be accessible by the\n  service account configured in this configuration (--service-account). If no\n  service account is defined, this image must be public.\n  '
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--container-predefined-image', choices={'codeoss': 'Code OSS', 'intellij': 'IntelliJ IDEA Ultimate', 'pycharm': 'PyCharm Professional', 'rider': 'Rider', 'webstorm': 'WebStorm', 'phpstorm': 'PhpStorm', 'rubymine': 'RubyMine', 'goland': 'GoLand', 'clion': 'CLion', 'base-image': 'Base image - no IDE', 'codeoss-cuda': 'Code OSS + CUDA toolkit'}, default='codeoss' if use_default else None, help=predefined_image_help_text)
    group.add_argument('--container-custom-image', type=str, help=custom_image_help_text)