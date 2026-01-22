from __future__ import (absolute_import, division, print_function)
class NetAppModule(object):
    """
    Common class for NetApp modules
    set of support functions to derive actions based
    on the current state of the system, and a desired state
    """

    def __init__(self):
        self.log = list()
        self.changed = False
        self.parameters = {'name': 'not intialized'}

    def set_parameters(self, ansible_params):
        self.parameters = dict()
        for param in ansible_params:
            if ansible_params[param] is not None:
                self.parameters[param] = ansible_params[param]
        return self.parameters

    def get_cd_action(self, current, desired):
        """ takes a desired state and a current state, and return an action:
            create, delete, None
            eg:
            is_present = 'absent'
            some_object = self.get_object(source)
            if some_object is not None:
                is_present = 'present'
            action = cd_action(current=is_present, desired = self.desired.state())
        """
        if 'state' in desired:
            desired_state = desired['state']
        else:
            desired_state = 'present'
        if current is None and desired_state == 'absent':
            return None
        if current is not None and desired_state == 'present':
            return None
        self.changed = True
        if current is not None:
            return 'delete'
        return 'create'

    def compare_and_update_values(self, current, desired, keys_to_compare):
        updated_values = dict()
        is_changed = False
        for key in keys_to_compare:
            if key in current:
                if key in desired and desired[key] is not None:
                    if current[key] != desired[key]:
                        updated_values[key] = desired[key]
                        is_changed = True
                    else:
                        updated_values[key] = current[key]
                else:
                    updated_values[key] = current[key]
        return (updated_values, is_changed)

    @staticmethod
    def check_keys(current, desired):
        """ TODO: raise an error if keys do not match
            with the exception of:
            new_name, state in desired
        """
        pass

    @staticmethod
    def compare_lists(current, desired, get_list_diff):
        """ compares two lists and return a list of elements that are either the desired elements or elements that are
            modified from the current state depending on the get_list_diff flag
            :param: current: current item attribute in ONTAP
            :param: desired: attributes from playbook
            :param: get_list_diff: specifies whether to have a diff of desired list w.r.t current list for an attribute
            :return: list of attributes to be modified
            :rtype: list
        """
        desired_diff_list = [item for item in desired if item not in current]
        current_diff_list = [item for item in current if item not in desired]
        if desired_diff_list or current_diff_list:
            if get_list_diff:
                return desired_diff_list
            else:
                return desired
        else:
            return []

    def get_modified_attributes(self, current, desired, get_list_diff=False, additional_keys=False):
        """ takes two dicts of attributes and return a dict of attributes that are
            not in the current state
            It is expected that all attributes of interest are listed in current and
            desired.
            The same assumption holds true for any nested directory.
            TODO: This is actually not true for the ElementSW 'attributes' directory.
                  Practically it means you cannot add or remove a key in a modify.
            :param: current: current attributes in ONTAP
            :param: desired: attributes from playbook
            :param: get_list_diff: specifies whether to have a diff of desired list w.r.t current list for an attribute
            :return: dict of attributes to be modified
            :rtype: dict

            NOTE: depending on the attribute, the caller may need to do a modify or a
            different operation (eg move volume if the modified attribute is an
            aggregate name)
        """
        modified = dict()
        if current is None:
            return modified
        self.check_keys(current, desired)
        for key, value in current.items():
            if key in desired and desired[key] is not None:
                if type(value) is list:
                    modified_list = self.compare_lists(value, desired[key], get_list_diff)
                    if modified_list:
                        modified[key] = modified_list
                elif type(value) is dict:
                    modified_dict = self.get_modified_attributes(value, desired[key], get_list_diff, additional_keys=True)
                    if modified_dict:
                        modified[key] = modified_dict
                elif cmp(value, desired[key]) != 0:
                    modified[key] = desired[key]
        if additional_keys:
            for key, value in desired.items():
                if key not in current:
                    modified[key] = desired[key]
        if modified:
            self.changed = True
        return modified

    def is_rename_action(self, source, target):
        """ takes a source and target object, and returns True
            if a rename is required
            eg:
            source = self.get_object(source_name)
            target = self.get_object(target_name)
            action = is_rename_action(source, target)
            :return: None for error, True for rename action, False otherwise
        """
        if source is None and target is None:
            return None
        if source is not None and target is not None:
            return False
        if source is None and target is not None:
            return False
        self.changed = True
        return True