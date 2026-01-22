from pathlib import Path
import sys
def __collect_members(self, fp):
    if fp.read(GLOBAL_HEADER_LENGTH) != GLOBAL_HEADER:
        raise ArError('Unable to find global header')
    while True:
        newmember = ArMember.from_file(fp, self.__fname, encoding=self.__encoding, errors=self.__errors)
        if not newmember:
            break
        self.__members.append(newmember)
        self.__members_dict[newmember.name] = newmember
        if newmember.size % 2 == 0:
            fp.seek(newmember.size, 1)
        else:
            fp.seek(newmember.size + 1, 1)