from saharaclient.api import base
class JobTypesManager(base.ResourceManager):
    resource_class = JobType

    def list(self, search_opts=None):
        """Get a list of job types supported by plugins."""
        query = base.get_query_string(search_opts)
        return self._list('/job-types%s' % query, 'job_types')