import boto
import boto.jsonresponse
from boto.compat import json
from boto.regioninfo import RegionInfo
from boto.connection import AWSQueryConnection
def update_configuration_template(self, application_name, template_name, description=None, option_settings=None, options_to_remove=None):
    """
        Updates the specified configuration template to have the
        specified properties or configuration option values.

        :type application_name: string
        :param application_name: The name of the application associated with
            the configuration template to update. If no application is found
            with this name, UpdateConfigurationTemplate returns an
            InvalidParameterValue error.

        :type template_name: string
        :param template_name: The name of the configuration template to update.
            If no configuration template is found with this name,
            UpdateConfigurationTemplate returns an InvalidParameterValue error.

        :type description: string
        :param description: A new description for the configuration.

        :type option_settings: list
        :param option_settings: A list of configuration option settings to
            update with the new specified option value.

        :type options_to_remove: list
        :param options_to_remove: A list of configuration options to remove
            from the configuration set.  Constraint: You can remove only
            UserDefined configuration options.

        :raises: InsufficientPrivilegesException
        """
    params = {'ApplicationName': application_name, 'TemplateName': template_name}
    if description:
        params['Description'] = description
    if option_settings:
        self._build_list_params(params, option_settings, 'OptionSettings.member', ('Namespace', 'OptionName', 'Value'))
    if options_to_remove:
        self.build_list_params(params, options_to_remove, 'OptionsToRemove.member')
    return self._get_response('UpdateConfigurationTemplate', params)