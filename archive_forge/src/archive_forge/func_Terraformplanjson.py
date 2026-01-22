from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
def Terraformplanjson(positional=True, required=True, help_text='Terraform plan in JSON format'):
    if positional:
        return base.Argument('terraform_plan_json', completer=FilesCompleter, help=help_text)
    else:
        return base.Argument('--terraform-plan-json', required=required, completer=FilesCompleter, help=help_text)