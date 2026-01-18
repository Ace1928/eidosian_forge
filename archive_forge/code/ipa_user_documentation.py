from __future__ import absolute_import, division, print_function
import base64
import hashlib
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native

    Return the public key fingerprint of a given public SSH key
    in format "[fp] [comment] (ssh-rsa)" where fp is of the format:
    FB:0C:AC:0A:07:94:5B:CE:75:6E:63:32:13:AD:AD:D7
    for md5 or
    SHA256:[base64]
    for sha256
    Comments are assumed to be all characters past the second
    whitespace character in the sshpubkey string.
    :param ssh_key:
    :param hash_algo:
    :return:
    