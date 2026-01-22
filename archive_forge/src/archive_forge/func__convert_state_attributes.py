import copy
import abc
import logging
import six
def _convert_state_attributes(self, state):
    label = state.get('label', state['name'])
    if self.machine.show_state_attributes:
        if 'tags' in state:
            label += ' [' + ', '.join(state['tags']) + ']'
        if 'on_enter' in state:
            label += '\\l- enter:\\l  + ' + '\\l  + '.join(state['on_enter'])
        if 'on_exit' in state:
            label += '\\l- exit:\\l  + ' + '\\l  + '.join(state['on_exit'])
        if 'timeout' in state:
            label += '\\l- timeout(' + state['timeout'] + 's) -> (' + ', '.join(state['on_timeout']) + ')'
    return label + '\\l'