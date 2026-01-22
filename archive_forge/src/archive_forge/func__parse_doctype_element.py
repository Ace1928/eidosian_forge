import re
def _parse_doctype_element(self, i, declstartpos):
    name, j = self._scan_name(i, declstartpos)
    if j == -1:
        return -1
    rawdata = self.rawdata
    if '>' in rawdata[j:]:
        return rawdata.find('>', j) + 1
    return -1