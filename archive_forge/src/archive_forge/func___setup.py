import sys
def __setup(self):
    if self.__lines:
        return
    data = None
    for filename in self.__filenames:
        try:
            with open(filename, encoding='utf-8') as fp:
                data = fp.read()
            break
        except OSError:
            pass
    if not data:
        data = self.__data
    self.__lines = data.split('\n')
    self.__linecnt = len(self.__lines)