import io
import math
import os
import typing
import weakref
def apply_redactions(page: fitz.Page, images: int=2, graphics: int=1, text: int=0) -> bool:
    """Apply the redaction annotations of the page.

    Args:
        page: the PDF page.
        images:
              0 - ignore images
              1 - remove all overlapping images
              2 - blank out overlapping image parts
              3 - remove image unless invisible
        graphics:
              0 - ignore graphics
              1 - remove graphics if contained in rectangle
              2 - remove all overlapping graphics
        text:
              0 - remove text
              1 - ignore text
    """

    def center_rect(annot_rect, new_text, font, fsize):
        """Calculate minimal sub-rectangle for the overlay text.

        Notes:
            Because 'insert_textbox' supports no vertical text centering,
            we calculate an approximate number of lines here and return a
            sub-rect with smaller height, which should still be sufficient.
        Args:
            annot_rect: the annotation rectangle
            new_text: the text to insert.
            font: the fontname. Must be one of the CJK or Base-14 set, else
                the rectangle is returned unchanged.
            fsize: the fontsize
        Returns:
            A rectangle to use instead of the annot rectangle.
        """
        exception_types = (ValueError, mupdf.FzErrorBase)
        if fitz.mupdf_version_tuple < (1, 24):
            exception_types = ValueError
        if not new_text:
            return annot_rect
        try:
            text_width = fitz.get_text_length(new_text, font, fsize)
        except exception_types:
            if g_exceptions_verbose:
                fitz.exception_info()
            return annot_rect
        line_height = fsize * 1.2
        limit = annot_rect.width
        h = math.ceil(text_width / limit) * line_height
        if h >= annot_rect.height:
            return annot_rect
        r = annot_rect
        y = (annot_rect.tl.y + annot_rect.bl.y - h) * 0.5
        r.y0 = y
        return r
    fitz.CheckParent(page)
    doc = page.parent
    if doc.is_encrypted or doc.is_closed:
        raise ValueError('document closed or encrypted')
    if not doc.is_pdf:
        raise ValueError('is no PDF')
    redact_annots = []
    for annot in page.annots(types=(fitz.PDF_ANNOT_REDACT,)):
        redact_annots.append(annot._get_redact_values())
    if redact_annots == []:
        return False
    rc = page._apply_redactions(text, images, graphics)
    if not rc:
        raise ValueError('Error applying redactions.')
    shape = page.new_shape()
    for redact in redact_annots:
        annot_rect = redact['rect']
        fill = redact['fill']
        if fill:
            shape.draw_rect(annot_rect)
            shape.finish(fill=fill, color=fill)
        if 'text' in redact.keys():
            new_text = redact['text']
            align = redact.get('align', 0)
            fname = redact['fontname']
            fsize = redact['fontsize']
            color = redact['text_color']
            trect = center_rect(annot_rect, new_text, fname, fsize)
            rc = -1
            while rc < 0 and fsize >= 4:
                rc = shape.insert_textbox(trect, new_text, fontname=fname, fontsize=fsize, color=color, align=align)
                fsize -= 0.5
    shape.commit()
    return True