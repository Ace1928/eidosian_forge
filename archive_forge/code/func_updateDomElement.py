from PySide2 import QtCore, QtGui, QtWidgets, QtXml
def updateDomElement(self, item, column):
    element = self.domElementForItem.get(id(item))
    if not element.isNull():
        if column == 0:
            oldTitleElement = element.firstChildElement('title')
            newTitleElement = self.domDocument.createElement('title')
            newTitleText = self.domDocument.createTextNode(item.text(0))
            newTitleElement.appendChild(newTitleText)
            element.replaceChild(newTitleElement, oldTitleElement)
        elif element.tagName() == 'bookmark':
            element.setAttribute('href', item.text(1))