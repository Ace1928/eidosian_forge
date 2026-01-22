import boto.vendored.regions.regions as _regions
class StaticEndpointBuilder(object):
    """Builds a static mapping of endpoints in the legacy format."""

    def __init__(self, resolver):
        """
        :type resolver: BotoEndpointResolver
        :param resolver: An endpoint resolver.
        """
        self._resolver = resolver

    def build_static_endpoints(self, service_names=None):
        """Build a set of static endpoints in the legacy boto2 format.

        :param service_names: The names of the services to build. They must
            use the names that boto2 uses, not boto3, e.g "ec2containerservice"
            and not "ecs". If no service names are provided, all available
            services will be built.

        :return: A dict consisting of::
            {"service": {"region": "full.host.name"}}
        """
        if service_names is None:
            service_names = self._resolver.get_available_services()
        static_endpoints = {}
        for name in service_names:
            endpoints_for_service = self._build_endpoints_for_service(name)
            if endpoints_for_service:
                static_endpoints[name] = endpoints_for_service
        self._handle_special_cases(static_endpoints)
        return static_endpoints

    def _build_endpoints_for_service(self, service_name):
        endpoints = {}
        regions = self._resolver.get_all_available_regions(service_name)
        for region_name in regions:
            endpoints[region_name] = self._resolver.resolve_hostname(service_name, region_name)
        return endpoints

    def _handle_special_cases(self, static_endpoints):
        if 'cloudsearch' in static_endpoints:
            cloudsearch_endpoints = static_endpoints['cloudsearch']
            static_endpoints['cloudsearchdomain'] = cloudsearch_endpoints