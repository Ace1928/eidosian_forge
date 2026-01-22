from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative

    ISBN Barcodes optionally print the EAN-5 supplemental price
    barcode (with the price in dollars or pounds). Set price to a string
    that follows the EAN-5 for ISBN spec:

        leading digit 0, 1 = GBP
                      3    = AUD
                      4    = NZD
                      5    = USD
                      6    = CAD
        next 4 digits = price between 00.00 and 99.98, i.e.:

        price='52499' # $24.99 USD
    