import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class UpdateStackTest(functional_base.FunctionalTestsBase):
    provider_template = {'heat_template_version': '2013-05-23', 'description': 'foo', 'resources': {'test1': {'type': 'My::TestResource'}}}
    provider_group_template = '\nheat_template_version: 2013-05-23\nparameters:\n  count:\n    type: number\n    default: 2\nresources:\n  test_group:\n    type: OS::Heat::ResourceGroup\n    properties:\n      count: {get_param: count}\n      resource_def:\n        type: My::TestResource\n'
    update_userdata_template = '\nheat_template_version: 2014-10-16\nparameters:\n  flavor:\n    type: string\n  user_data:\n    type: string\n  image:\n    type: string\n  network:\n    type: string\n\nresources:\n  server:\n    type: OS::Nova::Server\n    properties:\n      image: {get_param: image}\n      flavor: {get_param: flavor}\n      networks: [{network: {get_param: network} }]\n      user_data_format: SOFTWARE_CONFIG\n      user_data: {get_param: user_data}\n'
    fail_param_template = '\nheat_template_version: 2014-10-16\nparameters:\n  do_fail:\n    type: boolean\n    default: False\nresources:\n  aresource:\n    type: OS::Heat::TestResource\n    properties:\n      value: Test\n      fail: {get_param: do_fail}\n      wait_secs: 1\n'

    def test_stack_update_nochange(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_no_change'})
        stack_identifier = self.stack_create(template=template)
        expected_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(expected_resources, self.list_resources(stack_identifier))
        self.update_stack(stack_identifier, template)
        self.assertEqual(expected_resources, self.list_resources(stack_identifier))

    def test_stack_update_flavor_volume(self):
        parms = {'flavor': self.conf.minimal_instance_type, 'volume_size': 10, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name}
        stack_identifier = self.stack_create(template=test_template_updatae_flavor_and_volume_size, parameters=parms)
        parms_updated = parms
        parms_updated['volume_size'] = 20
        parms_updated['flavor'] = self.conf.instance_type
        self.update_stack(stack_identifier, template=test_template_updatae_flavor_and_volume_size, parameters=parms_updated)

    def test_stack_in_place_update(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_in_place'})
        stack_identifier = self.stack_create(template=template)
        expected_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(expected_resources, self.list_resources(stack_identifier))
        resource = self.client.resources.list(stack_identifier)
        initial_phy_id = resource[0].physical_resource_id
        tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_in_place_update'})
        self.update_stack(stack_identifier, tmpl_update)
        resource = self.client.resources.list(stack_identifier)
        self.assertEqual(initial_phy_id, resource[0].physical_resource_id)

    def test_stack_update_replace(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_replace'})
        stack_identifier = self.stack_create(template=template)
        expected_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(expected_resources, self.list_resources(stack_identifier))
        resource = self.client.resources.list(stack_identifier)
        initial_phy_id = resource[0].physical_resource_id
        tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_in_place_update', 'update_replace': True})
        self.update_stack(stack_identifier, tmpl_update)
        resource = self.client.resources.list(stack_identifier)
        self.assertNotEqual(initial_phy_id, resource[0].physical_resource_id)

    def test_stack_update_add_remove(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_add_remove'})
        stack_identifier = self.stack_create(template=template)
        initial_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_add_remove_update'})
        self.update_stack(stack_identifier, tmpl_update)
        updated_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
        self.assertEqual(updated_resources, self.list_resources(stack_identifier))
        self.update_stack(stack_identifier, template)
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))

    def test_stack_update_rollback(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_update_rollback'})
        stack_identifier = self.stack_create(template=template)
        initial_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_update_rollback', 'fail': True})
        self.update_stack(stack_identifier, tmpl_update, expected_status='ROLLBACK_COMPLETE', disable_rollback=False)
        updated_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(updated_resources, self.list_resources(stack_identifier))

    def test_stack_update_from_failed(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_update_failed'})
        stack_identifier = self.stack_create(template=template)
        initial_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'fail': True})
        self.update_stack(stack_identifier, tmpl_update, expected_status='UPDATE_FAILED')
        self.update_stack(stack_identifier, test_template_two_resource)
        updated_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
        self.assertEqual(updated_resources, self.list_resources(stack_identifier))

    @test.requires_convergence
    def test_stack_update_replace_manual_rollback(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'update_replace_value': '1'})
        stack_identifier = self.stack_create(template=template)
        original_resource_id = self.get_physical_resource_id(stack_identifier, 'test1')
        tmpl_update = _change_rsrc_properties(test_template_one_resource, ['test1'], {'update_replace_value': '2', 'fail': True})
        self.update_stack(stack_identifier, tmpl_update, expected_status='UPDATE_FAILED', disable_rollback=True)
        self.update_stack(stack_identifier, template)
        final_resource_id = self.get_physical_resource_id(stack_identifier, 'test1')
        self.assertEqual(original_resource_id, final_resource_id)

    def test_stack_update_provider(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_template'})
        files = {'provider.template': json.dumps(template)}
        env = {'resource_registry': {'My::TestResource': 'provider.template'}}
        stack_identifier = self.stack_create(template=self.provider_template, files=files, environment=env)
        initial_resources = {'test1': 'My::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test1')
        nested_id = nested_identifier.split('/')[-1]
        nested_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_provider_template'})
        files['provider.template'] = json.dumps(tmpl_update)
        self.update_stack(stack_identifier, self.provider_template, environment=env, files=files)
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        rsrc = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual(rsrc.physical_resource_id, nested_id)
        nested_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))

    def test_stack_update_alias_type(self):
        env = {'resource_registry': {'My::TestResource': 'OS::Heat::RandomString', 'My::TestResource2': 'OS::Heat::RandomString'}}
        stack_identifier = self.stack_create(template=self.provider_template, environment=env)
        p_res = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual('My::TestResource', p_res.resource_type)
        initial_resources = {'test1': 'My::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        res = self.client.resources.get(stack_identifier, 'test1')
        tmpl_update = copy.deepcopy(self.provider_template)
        tmpl_update['resources']['test1']['type'] = 'My::TestResource2'
        self.update_stack(stack_identifier, tmpl_update, environment=env)
        res_a = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual(res.physical_resource_id, res_a.physical_resource_id)
        self.assertEqual(res.attributes['value'], res_a.attributes['value'])

    def test_stack_update_alias_changes(self):
        env = {'resource_registry': {'My::TestResource': 'OS::Heat::RandomString'}}
        stack_identifier = self.stack_create(template=self.provider_template, environment=env)
        p_res = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual('My::TestResource', p_res.resource_type)
        initial_resources = {'test1': 'My::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        res = self.client.resources.get(stack_identifier, 'test1')
        env = {'resource_registry': {'My::TestResource': 'OS::Heat::TestResource'}}
        self.update_stack(stack_identifier, template=self.provider_template, environment=env)
        res_a = self.client.resources.get(stack_identifier, 'test1')
        self.assertNotEqual(res.physical_resource_id, res_a.physical_resource_id)

    def test_stack_update_provider_type(self):
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_template'})
        files = {'provider.template': json.dumps(template)}
        env = {'resource_registry': {'My::TestResource': 'provider.template', 'My::TestResource2': 'provider.template'}}
        stack_identifier = self.stack_create(template=self.provider_template, files=files, environment=env)
        p_res = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual('My::TestResource', p_res.resource_type)
        initial_resources = {'test1': 'My::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test1')
        nested_id = nested_identifier.split('/')[-1]
        nested_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        n_res = self.client.resources.get(nested_identifier, 'test1')
        tmpl_update = copy.deepcopy(self.provider_template)
        tmpl_update['resources']['test1']['type'] = 'My::TestResource2'
        self.update_stack(stack_identifier, tmpl_update, environment=env, files=files)
        p_res = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual('My::TestResource2', p_res.resource_type)
        self.assertEqual({u'test1': u'My::TestResource2'}, self.list_resources(stack_identifier))
        rsrc = self.client.resources.get(stack_identifier, 'test1')
        self.assertEqual(rsrc.physical_resource_id, nested_id)
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        n_res2 = self.client.resources.get(nested_identifier, 'test1')
        self.assertEqual(n_res.physical_resource_id, n_res2.physical_resource_id)

    def test_stack_update_provider_group(self):
        """Test two-level nested update."""
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_group_template'})
        files = {'provider.template': json.dumps(template)}
        env = {'resource_registry': {'My::TestResource': 'provider.template'}}
        stack_identifier = self.stack_create(template=self.provider_group_template, files=files, environment=env)
        initial_resources = {'test_group': 'OS::Heat::ResourceGroup'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test_group')
        nested_resources = {'0': 'My::TestResource', '1': 'My::TestResource'}
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        for n_rsrc in nested_resources:
            rsrc = self.client.resources.get(nested_identifier, n_rsrc)
            provider_stack = self.client.stacks.get(rsrc.physical_resource_id)
            provider_identifier = '%s/%s' % (provider_stack.stack_name, provider_stack.id)
            provider_resources = {u'test1': u'OS::Heat::TestResource'}
            self.assertEqual(provider_resources, self.list_resources(provider_identifier))
        tmpl_update = _change_rsrc_properties(test_template_two_resource, ['test1', 'test2'], {'value': 'test_provider_group_template'})
        files['provider.template'] = json.dumps(tmpl_update)
        self.update_stack(stack_identifier, self.provider_group_template, environment=env, files=files)
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_stack = self.client.stacks.get(nested_identifier)
        self.assertEqual('UPDATE_COMPLETE', nested_stack.stack_status)
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        for n_rsrc in nested_resources:
            rsrc = self.client.resources.get(nested_identifier, n_rsrc)
            provider_stack = self.client.stacks.get(rsrc.physical_resource_id)
            provider_identifier = '%s/%s' % (provider_stack.stack_name, provider_stack.id)
            provider_resources = {'test1': 'OS::Heat::TestResource', 'test2': 'OS::Heat::TestResource'}
            self.assertEqual(provider_resources, self.list_resources(provider_identifier))

    def test_stack_update_with_replacing_userdata(self):
        """Test case for updating userdata of instance.

        Confirm that we can update userdata of instance during updating stack
        by the user of member role.

        Make sure that a resource that inherits from StackUser can be deleted
        during updating stack.
        """
        if not self.conf.minimal_image_ref:
            raise self.skipException('No minimal image configured to test')
        if not self.conf.minimal_instance_type:
            raise self.skipException('No flavor configured to test')
        parms = {'flavor': self.conf.minimal_instance_type, 'image': self.conf.minimal_image_ref, 'network': self.conf.fixed_network_name, 'user_data': ''}
        stack_identifier = self.stack_create(template=self.update_userdata_template, parameters=parms)
        parms_updated = parms
        parms_updated['user_data'] = 'two'
        self.update_stack(stack_identifier, template=self.update_userdata_template, parameters=parms_updated)

    def test_stack_update_provider_group_patch(self):
        """Test two-level nested update with PATCH"""
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': 'test_provider_group_template'})
        files = {'provider.template': json.dumps(template)}
        env = {'resource_registry': {'My::TestResource': 'provider.template'}}
        stack_identifier = self.stack_create(template=self.provider_group_template, files=files, environment=env)
        initial_resources = {'test_group': 'OS::Heat::ResourceGroup'}
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_identifier = self.assert_resource_is_a_stack(stack_identifier, 'test_group')
        nested_resources = {'0': 'My::TestResource', '1': 'My::TestResource'}
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))
        params = {'count': 3}
        self.update_stack(stack_identifier, parameters=params, existing=True)
        self.assertEqual(initial_resources, self.list_resources(stack_identifier))
        nested_stack = self.client.stacks.get(nested_identifier)
        self.assertEqual('UPDATE_COMPLETE', nested_stack.stack_status)
        nested_resources['2'] = 'My::TestResource'
        self.assertEqual(nested_resources, self.list_resources(nested_identifier))

    def test_stack_update_from_failed_patch(self):
        """Test PATCH update from a failed state."""
        stack_identifier = self.stack_create(template='heat_template_version: 2014-10-16')
        self.update_stack(stack_identifier, template=self.fail_param_template, parameters={'do_fail': True}, expected_status='UPDATE_FAILED')
        self.update_stack(stack_identifier, parameters={'do_fail': False}, existing=True)
        self.assertEqual({u'aresource': u'OS::Heat::TestResource'}, self.list_resources(stack_identifier))

    def test_stack_update_with_new_env(self):
        """Update handles new resource types in the environment.

        If a resource type appears during an update and the update fails,
        retrying the update is able to find the type properly in the
        environment.
        """
        stack_identifier = self.stack_create(template=test_template_one_resource)
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'fail': True})
        template['resources']['test2'] = {'type': 'My::TestResource'}
        template['resources']['test1']['depends_on'] = 'test2'
        env = {'resource_registry': {'My::TestResource': 'OS::Heat::TestResource'}}
        self.update_stack(stack_identifier, template=template, environment=env, expected_status='UPDATE_FAILED')
        template = _change_rsrc_properties(template, ['test1'], {'fail': False})
        template['resources']['test2']['properties'] = {'action_wait_secs': {'update': 1}}
        self.update_stack(stack_identifier, template=template, environment=env)
        self.assertEqual({'test1': 'OS::Heat::TestResource', 'test2': 'My::TestResource'}, self.list_resources(stack_identifier))

    def test_stack_update_with_new_version(self):
        """Update handles new template version in failure.

        If a stack update fails while changing the template version, update is
        able to handle the new version fine.
        """
        stack_identifier = self.stack_create(template=test_template_one_resource)
        template = _change_rsrc_properties(test_template_two_resource, ['test1'], {'fail': True})
        template['heat_template_version'] = '2015-10-15'
        template['resources']['test2']['properties']['value'] = {'list_join': [',', ['a'], ['b']]}
        self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
        template = _change_rsrc_properties(template, ['test2'], {'value': 'Test2'})
        template['resources']['test1']['properties']['action_wait_secs'] = {'create': 1}
        self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
        self._stack_delete(stack_identifier)

    def test_stack_update_with_old_version(self):
        """Update handles old template version in failure.

        If a stack update fails while changing the template version, update is
        able to handle the old version fine.
        """
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': {'list_join': [',', ['a'], ['b']]}})
        template['heat_template_version'] = '2015-10-15'
        stack_identifier = self.stack_create(template=template)
        template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'fail': True})
        self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
        self._stack_delete(stack_identifier)

    def _test_conditional(self, test3_resource):
        """Update manages new conditions added.

        When a new resource is added during updates, the stacks handles the new
        conditions correctly, and doesn't fail to load them while the update is
        still in progress.
        """
        stack_identifier = self.stack_create(template=test_template_one_resource)
        updated_template = copy.deepcopy(test_template_two_resource)
        updated_template['conditions'] = {'cond1': True}
        updated_template['resources']['test3'] = test3_resource
        test2_props = updated_template['resources']['test2']['properties']
        test2_props['action_wait_secs'] = {'create': 30}
        self.update_stack(stack_identifier, template=updated_template, expected_status='UPDATE_IN_PROGRESS')

        def check_resources():

            def is_complete(r):
                return r.resource_status in {'CREATE_COMPLETE', 'UPDATE_COMPLETE'}
            resources = self.list_resources(stack_identifier, is_complete)
            if len(resources) < 2:
                return False
            self.assertIn('test3', resources)
            return True
        self.assertTrue(test.call_until_true(20, 2, check_resources))

    def test_stack_update_with_if_conditions(self):
        test3 = {'type': 'OS::Heat::TestResource', 'properties': {'value': {'if': ['cond1', 'val3', 'val4']}}}
        self._test_conditional(test3)

    def test_stack_update_with_conditions(self):
        test3 = {'type': 'OS::Heat::TestResource', 'condition': 'cond1', 'properties': {'value': 'foo'}}
        self._test_conditional(test3)

    def test_inplace_update_old_ref_deleted_failed_stack(self):
        template = '\nheat_template_version: rocky\nresources:\n  test1:\n    type: OS::Heat::TestResource\n    properties:\n      value: test\n  test2:\n    type: OS::Heat::TestResource\n    properties:\n      value: {get_attr: [test1, output]}\n  test3:\n    type: OS::Heat::TestResource\n    properties:\n      value: test3\n      fail: false\n      action_wait_secs:\n        update: 5\n'
        stack_identifier = self.stack_create(template=template)
        _template = template.replace('test1:', 'test-1:').replace('fail: false', 'fail: true')
        updated_template = _template.replace('{get_attr: [test1', '{get_attr: [test-1').replace('value: test3', 'value: test-3')
        self.update_stack(stack_identifier, template=updated_template, expected_status='UPDATE_FAILED')
        self.update_stack(stack_identifier, template=template, expected_status='UPDATE_COMPLETE')

    @test.requires_convergence
    def test_update_failed_changed_env_list_resources(self):
        template = {'heat_template_version': 'rocky', 'resources': {'test1': {'type': 'OS::Heat::TestResource', 'properties': {'value': 'foo'}}, 'my_res': {'type': 'My::TestResource', 'depends_on': 'test1'}, 'test2': {'depends_on': 'my_res', 'type': 'OS::Heat::TestResource'}}}
        env = {'resource_registry': {'My::TestResource': 'OS::Heat::TestResource'}}
        stack_identifier = self.stack_create(template=template, environment=env)
        update_template = copy.deepcopy(template)
        update_template['resources']['test1']['properties']['fail'] = 'true'
        update_template['resources']['test2']['depends_on'] = 'test1'
        del update_template['resources']['my_res']
        self.update_stack(stack_identifier, template=update_template, expected_status='UPDATE_FAILED')
        self.assertEqual(3, len(self.list_resources(stack_identifier)))