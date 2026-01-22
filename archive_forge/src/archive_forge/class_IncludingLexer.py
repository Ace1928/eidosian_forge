from fontTools.feaLib.error import FeatureLibError, IncludedFeaNotFound
from fontTools.feaLib.location import FeatureLibLocation
import re
import os
class IncludingLexer(object):
    """A Lexer that follows include statements.

    The OpenType feature file specification states that due to
    historical reasons, relative imports should be resolved in this
    order:

    1. If the source font is UFO format, then relative to the UFO's
       font directory
    2. relative to the top-level include file
    3. relative to the parent include file

    We only support 1 (via includeDir) and 2.
    """

    def __init__(self, featurefile, *, includeDir=None):
        """Initializes an IncludingLexer.

        Behavior:
            If includeDir is passed, it will be used to determine the top-level
            include directory to use for all encountered include statements. If it is
            not passed, ``os.path.dirname(featurefile)`` will be considered the
            include directory.
        """
        self.lexers_ = [self.make_lexer_(featurefile)]
        self.featurefilepath = self.lexers_[0].filename_
        self.includeDir = includeDir

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        while self.lexers_:
            lexer = self.lexers_[-1]
            try:
                token_type, token, location = next(lexer)
            except StopIteration:
                self.lexers_.pop()
                continue
            if token_type is Lexer.NAME and token == 'include':
                fname_type, fname_token, fname_location = lexer.next()
                if fname_type is not Lexer.FILENAME:
                    raise FeatureLibError('Expected file name', fname_location)
                if os.path.isabs(fname_token):
                    path = fname_token
                else:
                    if self.includeDir is not None:
                        curpath = self.includeDir
                    elif self.featurefilepath is not None:
                        curpath = os.path.dirname(self.featurefilepath)
                    else:
                        curpath = os.getcwd()
                    path = os.path.join(curpath, fname_token)
                if len(self.lexers_) >= 5:
                    raise FeatureLibError('Too many recursive includes', fname_location)
                try:
                    self.lexers_.append(self.make_lexer_(path))
                except FileNotFoundError as err:
                    raise IncludedFeaNotFound(fname_token, fname_location) from err
            else:
                return (token_type, token, location)
        raise StopIteration()

    @staticmethod
    def make_lexer_(file_or_path):
        if hasattr(file_or_path, 'read'):
            fileobj, closing = (file_or_path, False)
        else:
            filename, closing = (file_or_path, True)
            fileobj = open(filename, 'r', encoding='utf-8')
        data = fileobj.read()
        filename = getattr(fileobj, 'name', None)
        if closing:
            fileobj.close()
        return Lexer(data, filename)

    def scan_anonymous_block(self, tag):
        return self.lexers_[-1].scan_anonymous_block(tag)