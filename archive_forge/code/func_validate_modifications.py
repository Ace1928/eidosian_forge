from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def validate_modifications(ansible_to_ipa, module, ipa_otptoken, module_otptoken, unmodifiable_after_creation):
    """Checks to see if the requested modifications are valid.  Some elements
       cannot be modified after initial creation.  However, we still want to
       validate arguments that are specified, but are not different than what
       is currently set on the server.
    """
    modifications_valid = True
    for parameter in unmodifiable_after_creation:
        if ansible_to_ipa[parameter] in module_otptoken and ansible_to_ipa[parameter] in ipa_otptoken:
            mod_value = module_otptoken[ansible_to_ipa[parameter]]
            if parameter == 'otptype':
                ipa_value = ipa_otptoken[ansible_to_ipa[parameter]]
            else:
                if len(ipa_otptoken[ansible_to_ipa[parameter]]) != 1:
                    module.fail_json(msg='Invariant fail: Return value from IPA is not a list ' + 'of length 1.  Please open a bug report for the module.')
                if parameter == 'secretkey':
                    mod_value = base32_to_base64(mod_value)
                    ipa_value = ipa_otptoken[ansible_to_ipa[parameter]][0]['__base64__']
                    if '__base64__' in ipa_otptoken[ansible_to_ipa[parameter]][0]:
                        ipa_value = ipa_otptoken[ansible_to_ipa[parameter]][0]['__base64__']
                    elif '__base32__' in ipa_otptoken[ansible_to_ipa[parameter]][0]:
                        b32key = ipa_otptoken[ansible_to_ipa[parameter]][0]['__base32__']
                        b64key = base32_to_base64(b32key)
                        ipa_value = b64key
                    else:
                        ipa_value = None
                else:
                    ipa_value = ipa_otptoken[ansible_to_ipa[parameter]][0]
            if mod_value != ipa_value:
                modifications_valid = False
                fail_message = "Parameter '" + parameter + "' cannot be changed once " + 'the OTP is created and the requested value specified here (' + str(mod_value) + ') differs from what is set in the IPA server (' + str(ipa_value) + ')'
                module.fail_json(msg=fail_message)
    return modifications_valid