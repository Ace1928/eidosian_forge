from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text, to_native
from subprocess import Popen, PIPE
class Sops:
    """ Utility class to perform sops CLI actions """

    @staticmethod
    def _add_options(command, env, get_option_value, options):
        if get_option_value is None:
            return
        for option, f in options.items():
            v = get_option_value(option)
            if v is not None:
                f(v, command, env)

    @staticmethod
    def get_sops_binary(get_option_value):
        cmd = get_option_value('sops_binary') if get_option_value else None
        if cmd is None:
            cmd = 'sops'
        return cmd

    @staticmethod
    def decrypt(encrypted_file, content=None, display=None, decode_output=True, rstrip=True, input_type=None, output_type=None, get_option_value=None, module=None):
        command = [Sops.get_sops_binary(get_option_value)]
        env = os.environ.copy()
        Sops._add_options(command, env, get_option_value, GENERAL_OPTIONS)
        if input_type is not None:
            command.extend(['--input-type', input_type])
        if output_type is not None:
            command.extend(['--output-type', output_type])
        if content is not None:
            encrypted_file = '/dev/stdin'
        command.extend(['--decrypt', encrypted_file])
        if module:
            exit_code, output, err = module.run_command(command, environ_update=env, encoding=None, data=content, binary_data=True)
        else:
            process = Popen(command, stdin=None if content is None else PIPE, stdout=PIPE, stderr=PIPE, env=env)
            output, err = process.communicate(input=content)
            exit_code = process.returncode
        if decode_output:
            output = to_text(output, errors='surrogate_or_strict')
        if err and display:
            display.vvvv(to_text(err, errors='surrogate_or_strict'))
        if exit_code != 0:
            raise SopsError(encrypted_file, exit_code, err, decryption=True)
        if rstrip:
            output = output.rstrip()
        return output

    @staticmethod
    def encrypt(data, display=None, cwd=None, input_type=None, output_type=None, get_option_value=None, module=None):
        command = [Sops.get_sops_binary(get_option_value)]
        env = os.environ.copy()
        Sops._add_options(command, env, get_option_value, GENERAL_OPTIONS)
        Sops._add_options(command, env, get_option_value, ENCRYPT_OPTIONS)
        if input_type is not None:
            command.extend(['--input-type', input_type])
        if output_type is not None:
            command.extend(['--output-type', output_type])
        command.extend(['--encrypt', '/dev/stdin'])
        if module:
            exit_code, output, err = module.run_command(command, data=data, binary_data=True, cwd=cwd, environ_update=env, encoding=None)
        else:
            process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=cwd, env=env)
            output, err = process.communicate(input=data)
            exit_code = process.returncode
        if err and display:
            display.vvvv(to_text(err, errors='surrogate_or_strict'))
        if exit_code != 0:
            raise SopsError('to stdout', exit_code, err, decryption=False)
        return output