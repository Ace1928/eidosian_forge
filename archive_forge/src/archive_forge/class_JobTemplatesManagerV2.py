from saharaclient.api import base
class JobTemplatesManagerV2(base.ResourceManager):
    resource_class = JobTemplate
    NotUpdated = base.NotUpdated()

    def create(self, name, type, mains=None, libs=None, description=None, interface=None, is_public=None, is_protected=None):
        """Create a Job Template."""
        data = {'name': name, 'type': type}
        self._copy_if_defined(data, description=description, mains=mains, libs=libs, interface=interface, is_public=is_public, is_protected=is_protected)
        return self._create('/%s' % 'job-templates', data, 'job_template')

    def list(self, search_opts=None, limit=None, marker=None, sort_by=None, reverse=None):
        """Get a list of Job Templates."""
        query = base.get_query_string(search_opts, limit=limit, marker=marker, sort_by=sort_by, reverse=reverse)
        url = '/%s%s' % ('job-templates', query)
        return self._page(url, 'job_templates', limit)

    def get(self, job_id):
        """Get information about a Job Template."""
        return self._get('/%s/%s' % ('job-templates', job_id), 'job_template')

    def get_configs(self, job_type):
        """Get config hints for a specified Job Template type."""
        return self._get('/%s/config-hints/%s' % ('job-templates', job_type))

    def delete(self, job_id):
        """Delete a Job Template."""
        self._delete('/%s/%s' % ('job-templates', job_id))

    def update(self, job_id, name=NotUpdated, description=NotUpdated, is_public=NotUpdated, is_protected=NotUpdated):
        """Update a Job Template."""
        data = {}
        self._copy_if_updated(data, name=name, description=description, is_public=is_public, is_protected=is_protected)
        return self._patch('/%s/%s' % ('job-templates', job_id), data)