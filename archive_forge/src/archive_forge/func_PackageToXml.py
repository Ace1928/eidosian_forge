import time
from xml.etree.ElementTree import Element, ElementTree, SubElement
def PackageToXml(pkg, summary='N/A', trainingDataId='N/A', dataPerformance=[], recommendedThreshold=None, classDescriptions=None, modelType=None, modelOrganism=None):
    """ generates XML for a package that follows the RD_Model.dtd

    If provided, dataPerformance should be a sequence of 2-tuples:
      ( note, performance )
    where performance is of the form:
      ( accuracy, avgCorrectConf, avgIncorrectConf, confusionMatrix, thresh, avgSkipConf, nSkipped )
      the last four elements are optional

    """
    head = Element('RDModelInfo')
    name = SubElement(head, 'ModelName')
    notes = pkg.GetNotes()
    if not notes:
        notes = 'Unnamed model'
    name.text = notes
    summ = SubElement(head, 'ModelSummary')
    summ.text = summary
    calc = pkg.GetCalculator()
    descrs = SubElement(head, 'ModelDescriptors')
    for name, summary, func in zip(calc.GetDescriptorNames(), calc.GetDescriptorSummaries(), calc.GetDescriptorFuncs()):
        descr = SubElement(descrs, 'Descriptor')
        elem = SubElement(descr, 'DescriptorName')
        elem.text = name
        elem = SubElement(descr, 'DescriptorDetail')
        elem.text = summary
        if hasattr(func, 'version'):
            vers = SubElement(descr, 'DescriptorVersion')
            major, minor, patch = func.version.split('.')
            elem = SubElement(vers, 'VersionMajor')
            elem.text = major
            elem = SubElement(vers, 'VersionMinor')
            elem.text = minor
            elem = SubElement(vers, 'VersionPatch')
            elem.text = patch
    elem = SubElement(head, 'TrainingDataId')
    elem.text = trainingDataId
    for description, perfData in dataPerformance:
        dataNode = SubElement(head, 'ValidationData')
        note = SubElement(dataNode, 'ScreenNote')
        note.text = description
        perf = SubElement(dataNode, 'PerformanceData')
        _ConvertModelPerformance(perf, perfData)
    if recommendedThreshold:
        elem = SubElement(head, 'RecommendedThreshold')
        elem.text = str(recommendedThreshold)
    if classDescriptions:
        elem = SubElement(head, 'ClassDescriptions')
        for val, text in classDescriptions:
            descr = SubElement(elem, 'ClassDescription')
            valElem = SubElement(descr, 'ClassVal')
            valElem.text = str(val)
            valText = SubElement(descr, 'ClassText')
            valText.text = str(text)
    if modelType:
        elem = SubElement(head, 'ModelType')
        elem.text = modelType
    if modelOrganism:
        elem = SubElement(head, 'ModelOrganism')
        elem.text = modelOrganism
    hist = SubElement(head, 'ModelHistory')
    revision = SubElement(hist, 'Revision')
    tm = time.localtime()
    date = SubElement(revision, 'RevisionDate')
    elem = SubElement(date, 'Year')
    elem.text = str(tm[0])
    elem = SubElement(date, 'Month')
    elem.text = str(tm[1])
    elem = SubElement(date, 'Day')
    elem.text = str(tm[2])
    note = SubElement(revision, 'RevisionNote')
    note.text = 'Created'
    return ElementTree(head)