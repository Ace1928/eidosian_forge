from octavia_lib.api.drivers import exceptions
def validate_flavor(self, flavor_metadata):
    """Validates if driver can support the flavor.

        :param flavor_metadata: Dictionary with flavor metadata.
        :type flavor_metadata: dict
        :return: Nothing if the flavor is valid and supported.
        :raises DriverError: An unexpected error occurred in the driver.
        :raises NotImplementedError: The driver does not support flavors.
        :raises UnsupportedOptionError: if driver does not
          support one of the configuration options.
        :raises octavia_lib.api.drivers.exceptions.NotFound: if the driver
          cannot find a resource.
        """
    raise exceptions.NotImplementedError(user_fault_string='This provider does not support validating flavors.', operator_fault_string='This provider does not support validating the supported flavor metadata.')