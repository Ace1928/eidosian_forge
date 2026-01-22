import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
class EndpointResolver(BaseEndpointResolver):
    """Resolves endpoints based on partition endpoint metadata"""

    def __init__(self, endpoint_data):
        """
        :param endpoint_data: A dict of partition data.
        """
        if 'partitions' not in endpoint_data:
            raise ValueError('Missing "partitions" in endpoint data')
        self._endpoint_data = endpoint_data

    def get_available_partitions(self):
        result = []
        for partition in self._endpoint_data['partitions']:
            result.append(partition['partition'])
        return result

    def get_available_endpoints(self, service_name, partition_name='aws', allow_non_regional=False):
        result = []
        for partition in self._endpoint_data['partitions']:
            if partition['partition'] != partition_name:
                continue
            services = partition['services']
            if service_name not in services:
                continue
            for endpoint_name in services[service_name]['endpoints']:
                if allow_non_regional or endpoint_name in partition['regions']:
                    result.append(endpoint_name)
        return result

    def construct_endpoint(self, service_name, region_name=None):
        for partition in self._endpoint_data['partitions']:
            result = self._endpoint_for_partition(partition, service_name, region_name)
            if result:
                return result

    def _endpoint_for_partition(self, partition, service_name, region_name):
        service_data = partition['services'].get(service_name, DEFAULT_SERVICE_DATA)
        if region_name is None:
            if 'partitionEndpoint' in service_data:
                region_name = service_data['partitionEndpoint']
            else:
                raise NoRegionError()
        if region_name in service_data['endpoints']:
            return self._resolve(partition, service_name, service_data, region_name)
        if self._region_match(partition, region_name):
            partition_endpoint = service_data.get('partitionEndpoint')
            is_regionalized = service_data.get('isRegionalized', True)
            if partition_endpoint and (not is_regionalized):
                LOG.debug('Using partition endpoint for %s, %s: %s', service_name, region_name, partition_endpoint)
                return self._resolve(partition, service_name, service_data, partition_endpoint)
            LOG.debug('Creating a regex based endpoint for %s, %s', service_name, region_name)
            return self._resolve(partition, service_name, service_data, region_name)

    def _region_match(self, partition, region_name):
        if region_name in partition['regions']:
            return True
        if 'regionRegex' in partition:
            return re.compile(partition['regionRegex']).match(region_name)
        return False

    def _resolve(self, partition, service_name, service_data, endpoint_name):
        result = service_data['endpoints'].get(endpoint_name, {})
        result['partition'] = partition['partition']
        result['endpointName'] = endpoint_name
        self._merge_keys(service_data.get('defaults', {}), result)
        self._merge_keys(partition.get('defaults', {}), result)
        hostname = result.get('hostname', DEFAULT_URI_TEMPLATE)
        result['hostname'] = self._expand_template(partition, result['hostname'], service_name, endpoint_name)
        if 'sslCommonName' in result:
            result['sslCommonName'] = self._expand_template(partition, result['sslCommonName'], service_name, endpoint_name)
        result['dnsSuffix'] = partition['dnsSuffix']
        return result

    def _merge_keys(self, from_data, result):
        for key in from_data:
            if key not in result:
                result[key] = from_data[key]

    def _expand_template(self, partition, template, service_name, endpoint_name):
        return template.format(service=service_name, region=endpoint_name, dnsSuffix=partition['dnsSuffix'])