from breezy import commands, osutils, tests, trace, ui
from breezy.tests import script
class cmd_test_confirm(commands.Command):

    def run(self):
        if ui.ui_factory.get_boolean('Really do it'):
            self.outf.write('Do it!\n')
        else:
            print('ok, no')