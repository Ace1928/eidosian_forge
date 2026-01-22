from distutils.command import build_py
class test_command(build_py.build_py):
    command_name = 'build_py'

    def run(self):
        print('Running custom build_py command.')
        return build_py.build_py.run(self)