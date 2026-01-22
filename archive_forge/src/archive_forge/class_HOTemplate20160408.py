import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
class HOTemplate20160408(HOTemplate20151015):
    functions = {'get_attr': hot_funcs.GetAttAllAttributes, 'get_file': hot_funcs.GetFile, 'get_param': hot_funcs.GetParam, 'get_resource': hot_funcs.GetResource, 'list_join': hot_funcs.JoinMultiple, 'repeat': hot_funcs.Repeat, 'resource_facade': hot_funcs.ResourceFacade, 'str_replace': hot_funcs.ReplaceJson, 'digest': hot_funcs.Digest, 'str_split': hot_funcs.StrSplit, 'map_merge': hot_funcs.MapMerge, 'Fn::Select': hot_funcs.Removed, 'Fn::GetAZs': hot_funcs.Removed, 'Fn::Join': hot_funcs.Removed, 'Fn::Split': hot_funcs.Removed, 'Fn::Replace': hot_funcs.Removed, 'Fn::Base64': hot_funcs.Removed, 'Fn::MemberListToMap': hot_funcs.Removed, 'Fn::ResourceFacade': hot_funcs.Removed, 'Ref': hot_funcs.Removed}