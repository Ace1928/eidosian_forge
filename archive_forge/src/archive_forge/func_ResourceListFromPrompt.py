from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core.console import console_io
def ResourceListFromPrompt(name, list_func, end_empty_message=None):
    """Returns a list of resources selected by the user.

  Args:
    name: the entity name for the resources being selected.
    list_func: a zero argument function that will return a list of existing
      resources.
    end_empty_message: text for the menu option that will return an empty list.
  """
    resource_list = list_func()
    if not resource_list:
        docs_name = resource_args.ENTITIES[name].docs_name
        raise errors.EntityNotFoundError(message='Could not find any %s to select. Check that at least one %s has been created and is properly configued for use.' % (docs_name, docs_name))
    chosen = []
    available = None
    menu_option = len(resource_list) + 1
    while menu_option != len(resource_list):
        if menu_option < len(chosen):
            chosen = chosen[:menu_option] + chosen[menu_option + 1:]
        elif menu_option < len(resource_list):
            index = menu_option - len(chosen)
            chosen.append(available[index])
        available = [item for item in resource_list if item not in chosen]
        options = ['Remove `%s`' % item for item in chosen]
        options += ['Add `%s`' % item for item in available]
        if chosen:
            message = 'Currently selected: %s' % ', '.join(chosen)
            options.append('Done')
        else:
            message = 'No %s selected yet' % resource_args.ENTITIES[name].docs_name
            if end_empty_message is not None:
                options.append(end_empty_message)
        menu_option = console_io.PromptChoice(options, message=message)
    return chosen