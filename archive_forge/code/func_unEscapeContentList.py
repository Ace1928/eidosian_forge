def unEscapeContentList(contentList):
    result = []
    for e in contentList:
        if '&' in e:
            for old, new in replacelist:
                e = e.replace(old, new)
        result.append(e)
    return result