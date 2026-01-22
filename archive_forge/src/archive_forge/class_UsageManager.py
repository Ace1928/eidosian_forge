import oslo_utils
from novaclient import api_versions
from novaclient import base
class UsageManager(base.ManagerWithFind):
    """
    Manage :class:`Usage` resources.
    """
    resource_class = Usage
    usage_prefix = 'os-simple-tenant-usage'

    def _usage_query(self, start, end, marker=None, limit=None, detailed=None):
        query = '?start=%s&end=%s' % (start.isoformat(), end.isoformat())
        if limit:
            query = '%s&limit=%s' % (query, int(limit))
        if marker:
            query = '%s&marker=%s' % (query, marker)
        if detailed is not None:
            query = '%s&detailed=%s' % (query, int(bool(detailed)))
        return query

    @api_versions.wraps('2.0', '2.39')
    def list(self, start, end, detailed=False):
        """
        Get usage for all tenants

        :param start: :class:`datetime.datetime` Start date in UTC
        :param end: :class:`datetime.datetime` End date in UTC
        :param detailed: Whether to include information about each
                         instance whose usage is part of the report
        :rtype: list of :class:`Usage`.
        """
        query_string = self._usage_query(start, end, detailed=detailed)
        url = '/%s%s' % (self.usage_prefix, query_string)
        return self._list(url, 'tenant_usages')

    @api_versions.wraps('2.40')
    def list(self, start, end, detailed=False, marker=None, limit=None):
        """
        Get usage for all tenants

        :param start: :class:`datetime.datetime` Start date in UTC
        :param end: :class:`datetime.datetime` End date in UTC
        :param detailed: Whether to include information about each
                         instance whose usage is part of the report
        :param marker: Begin returning usage data for instances that appear
                       later in the instance list than that represented by
                       this instance UUID (optional).
        :param limit: Maximum number of instances to include in the usage
                      (optional). Note the API server has a configurable
                      default limit. If no limit is specified here or limit
                      is larger than default, the default limit will be used.
        :rtype: list of :class:`Usage`.
        """
        query_string = self._usage_query(start, end, marker, limit, detailed)
        url = '/%s%s' % (self.usage_prefix, query_string)
        return self._list(url, 'tenant_usages')

    @api_versions.wraps('2.0', '2.39')
    def get(self, tenant_id, start, end):
        """
        Get usage for a specific tenant.

        :param tenant_id: Tenant ID to fetch usage for
        :param start: :class:`datetime.datetime` Start date in UTC
        :param end: :class:`datetime.datetime` End date in UTC
        :rtype: :class:`Usage`
        """
        query_string = self._usage_query(start, end)
        url = '/%s/%s%s' % (self.usage_prefix, tenant_id, query_string)
        return self._get(url, 'tenant_usage')

    @api_versions.wraps('2.40')
    def get(self, tenant_id, start, end, marker=None, limit=None):
        """
        Get usage for a specific tenant.

        :param tenant_id: Tenant ID to fetch usage for
        :param start: :class:`datetime.datetime` Start date in UTC
        :param end: :class:`datetime.datetime` End date in UTC
        :param marker: Begin returning usage data for instances that appear
                       later in the instance list than that represented by
                       this instance UUID (optional).
        :param limit: Maximum number of instances to include in the usage
                      (optional). Note the API server has a configurable
                      default limit. If no limit is specified here or limit
                      is larger than default, the default limit will be used.
        :rtype: :class:`Usage`
        """
        query_string = self._usage_query(start, end, marker, limit)
        url = '/%s/%s%s' % (self.usage_prefix, tenant_id, query_string)
        return self._get(url, 'tenant_usage')