import inspect
import re
import six
def ui_complete_ls(self, parameters, text, current_param):
    """
        Parameter auto-completion method for user command ls.
        @param parameters: Parameters on the command line.
        @type parameters: dict
        @param text: Current text of parameter being typed by the user.
        @type text: str
        @param current_param: Name of parameter to complete.
        @type current_param: str
        @return: Possible completions
        @rtype: list of str
        """
    if current_param == 'path':
        basedir, slash, partial_name = text.rpartition('/')
        basedir = basedir + slash
        target = self.get_node(basedir)
        names = [child.name for child in target.children]
        completions = []
        for name in names:
            num_matches = 0
            if name.startswith(partial_name):
                num_matches += 1
                if num_matches == 1:
                    completions.append('%s%s/' % (basedir, name))
                else:
                    completions.append('%s%s' % (basedir, name))
        if len(completions) == 1:
            if not self.get_node(completions[0]).children:
                completions[0] = completions[0].rstrip('/') + ' '
        bookmarks = ['@' + bookmark for bookmark in self.shell.prefs['bookmarks'] if ('@' + bookmark).startswith(text)]
        self.shell.log.debug('Found bookmarks %s.' % str(bookmarks))
        if bookmarks:
            completions.extend(bookmarks)
        self.shell.log.debug('Completions are %s.' % str(completions))
        return completions
    elif current_param == 'depth':
        if text:
            try:
                int(text.strip())
            except ValueError:
                self.shell.log.debug('Text is not a number.')
                return []
        return [text + number for number in [str(num) for num in range(10)] if (text + number).startswith(text)]