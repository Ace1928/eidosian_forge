import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_params
from heat.engine import function
from heat.engine import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
class HeatTemplate(CfnTemplateBase):
    functions = {'Fn::FindInMap': cfn_funcs.FindInMap, 'Fn::GetAZs': cfn_funcs.GetAZs, 'Ref': cfn_funcs.Ref, 'Fn::GetAtt': cfn_funcs.GetAtt, 'Fn::Select': cfn_funcs.Select, 'Fn::Join': cfn_funcs.Join, 'Fn::Split': cfn_funcs.Split, 'Fn::Replace': cfn_funcs.Replace, 'Fn::Base64': cfn_funcs.Base64, 'Fn::MemberListToMap': cfn_funcs.MemberListToMap, 'Fn::ResourceFacade': cfn_funcs.ResourceFacade}