import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
def zsh_completion_function(out, function_name='_brz', debug=False, no_plugins=False, selected_plugins=None):
    dc = DataCollector(no_plugins=no_plugins, selected_plugins=selected_plugins)
    data = dc.collect()
    cg = ZshCodeGen(data, function_name=function_name, debug=debug)
    res = cg.script()
    out.write(res)