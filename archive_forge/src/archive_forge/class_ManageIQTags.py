from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
class ManageIQTags(object):
    """
        Object to execute tags management operations of manageiq resources.
    """

    def __init__(self, manageiq, resource_type, resource_id):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.resource_url = '{api_url}/{resource_type}/{resource_id}'.format(api_url=self.api_url, resource_type=resource_type, resource_id=resource_id)

    def full_tag_name(self, tag):
        """ Returns the full tag name in manageiq
        """
        return '/managed/{tag_category}/{tag_name}'.format(tag_category=tag['category'], tag_name=tag['name'])

    def clean_tag_object(self, tag):
        """ Clean a tag object to have human readable form of:
        {
            full_name: STR,
            name: STR,
            display_name: STR,
            category: STR
        }
        """
        full_name = tag.get('name')
        categorization = tag.get('categorization', {})
        return dict(full_name=full_name, name=categorization.get('name'), display_name=categorization.get('display_name'), category=categorization.get('category', {}).get('name'))

    def query_resource_tags(self):
        """ Returns a set of the tag objects assigned to the resource
        """
        url = '{resource_url}/tags?expand=resources&attributes=categorization'
        try:
            response = self.client.get(url.format(resource_url=self.resource_url))
        except Exception as e:
            msg = 'Failed to query {resource_type} tags: {error}'.format(resource_type=self.resource_type, error=e)
            self.module.fail_json(msg=msg)
        resources = response.get('resources', [])
        tags = [self.clean_tag_object(tag) for tag in resources]
        return tags

    def tags_to_update(self, tags, action):
        """ Create a list of tags we need to update in ManageIQ.

        Returns:
            Whether or not a change took place and a message describing the
            operation executed.
        """
        tags_to_post = []
        assigned_tags = self.query_resource_tags()
        assigned_tags_set = set([tag['full_name'] for tag in assigned_tags])
        for tag in tags:
            assigned = self.full_tag_name(tag) in assigned_tags_set
            if assigned and action == 'unassign':
                tags_to_post.append(tag)
            elif not assigned and action == 'assign':
                tags_to_post.append(tag)
        return tags_to_post

    def assign_or_unassign_tags(self, tags, action):
        """ Perform assign/unassign action
        """
        tags_to_post = self.tags_to_update(tags, action)
        if not tags_to_post:
            return dict(changed=False, msg='Tags already {action}ed, nothing to do'.format(action=action))
        url = '{resource_url}/tags'.format(resource_url=self.resource_url)
        try:
            response = self.client.post(url, action=action, resources=tags)
        except Exception as e:
            msg = 'Failed to {action} tag: {error}'.format(action=action, error=e)
            self.module.fail_json(msg=msg)
        for result in response['results']:
            if not result['success']:
                msg = 'Failed to {action}: {message}'.format(action=action, message=result['message'])
                self.module.fail_json(msg=msg)
        return dict(changed=True, msg='Successfully {action}ed tags'.format(action=action))