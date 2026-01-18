import inspect
import re
import six
def ui_command_pwd(self):
    """
        Displays the current path.

        SEE ALSO
        ========
        ls cd
        """
    self.shell.con.display(self.path)