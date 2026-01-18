import inspect
import re
import six
def ui_command_bookmarks(self, action, bookmark=None):
    """
        Manage your bookmarks.

        Note that you can also access your bookmarks with the
        `cd` command. For instance, the following commands
        are equivalent:
            - `cd @mybookmark`
            - `bookmarks go mybookmark``

        You can also use bookmarks anywhere where you would use
        a normal path:
            - `@mybookmark ls` performs the `ls` command
            in the bookmarked path.
            - `ls @mybookmark` shows you the objects tree from
            the bookmarked path.


        PARAMETERS
        ==========

        action
        ------
        The "action" parameter is one of:
            - `add` adds the current path to your bookmarks.
            - `del` deletes a bookmark.
            - `go` takes you to a bookmarked path.
            - `show` shows you all your bookmarks.

        bookmark
        --------
        This is the name of the bookmark.

        SEE ALSO
        ========
        ls cd
        """
    if action == 'add' and bookmark:
        if bookmark in self.shell.prefs['bookmarks']:
            raise ExecutionError('Bookmark %s already exists.' % bookmark)
        self.shell.prefs['bookmarks'][bookmark] = self.path
        self.shell.prefs.save()
        self.shell.log.info('Bookmarked %s as %s.' % (self.path, bookmark))
    elif action == 'del' and bookmark:
        if bookmark not in self.shell.prefs['bookmarks']:
            raise ExecutionError('No such bookmark %s.' % bookmark)
        del self.shell.prefs['bookmarks'][bookmark]
        self.shell.prefs.save()
        self.shell.log.info('Deleted bookmark %s.' % bookmark)
    elif action == 'go' and bookmark:
        if bookmark not in self.shell.prefs['bookmarks']:
            raise ExecutionError('No such bookmark %s.' % bookmark)
        return self.ui_command_cd(self.shell.prefs['bookmarks'][bookmark])
    elif action == 'show':
        bookmarks = self.shell.con.dedent('\n                                              BOOKMARKS\n                                              =========\n\n                                              ')
        if not self.shell.prefs['bookmarks']:
            bookmarks += 'No bookmarks yet.\n'
        else:
            for bookmark, path in six.iteritems(self.shell.prefs['bookmarks']):
                if len(bookmark) == 1:
                    bookmark += '\x00'
                underline = ''.ljust(len(bookmark), '-')
                bookmarks += '%s\n%s\n%s\n\n' % (bookmark, underline, path)
        self.shell.con.epy_write(bookmarks)
    else:
        raise ExecutionError("Syntax error, see 'help bookmarks'.")