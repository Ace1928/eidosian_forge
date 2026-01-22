from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
class pdf_document(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    super = property(_mupdf.pdf_document_super_get, _mupdf.pdf_document_super_set)
    file = property(_mupdf.pdf_document_file_get, _mupdf.pdf_document_file_set)
    version = property(_mupdf.pdf_document_version_get, _mupdf.pdf_document_version_set)
    is_fdf = property(_mupdf.pdf_document_is_fdf_get, _mupdf.pdf_document_is_fdf_set)
    startxref = property(_mupdf.pdf_document_startxref_get, _mupdf.pdf_document_startxref_set)
    file_size = property(_mupdf.pdf_document_file_size_get, _mupdf.pdf_document_file_size_set)
    crypt = property(_mupdf.pdf_document_crypt_get, _mupdf.pdf_document_crypt_set)
    ocg = property(_mupdf.pdf_document_ocg_get, _mupdf.pdf_document_ocg_set)
    oi = property(_mupdf.pdf_document_oi_get, _mupdf.pdf_document_oi_set)
    max_xref_len = property(_mupdf.pdf_document_max_xref_len_get, _mupdf.pdf_document_max_xref_len_set)
    num_xref_sections = property(_mupdf.pdf_document_num_xref_sections_get, _mupdf.pdf_document_num_xref_sections_set)
    saved_num_xref_sections = property(_mupdf.pdf_document_saved_num_xref_sections_get, _mupdf.pdf_document_saved_num_xref_sections_set)
    num_incremental_sections = property(_mupdf.pdf_document_num_incremental_sections_get, _mupdf.pdf_document_num_incremental_sections_set)
    xref_base = property(_mupdf.pdf_document_xref_base_get, _mupdf.pdf_document_xref_base_set)
    disallow_new_increments = property(_mupdf.pdf_document_disallow_new_increments_get, _mupdf.pdf_document_disallow_new_increments_set)
    local_xref = property(_mupdf.pdf_document_local_xref_get, _mupdf.pdf_document_local_xref_set)
    local_xref_nesting = property(_mupdf.pdf_document_local_xref_nesting_get, _mupdf.pdf_document_local_xref_nesting_set)
    xref_sections = property(_mupdf.pdf_document_xref_sections_get, _mupdf.pdf_document_xref_sections_set)
    saved_xref_sections = property(_mupdf.pdf_document_saved_xref_sections_get, _mupdf.pdf_document_saved_xref_sections_set)
    xref_index = property(_mupdf.pdf_document_xref_index_get, _mupdf.pdf_document_xref_index_set)
    save_in_progress = property(_mupdf.pdf_document_save_in_progress_get, _mupdf.pdf_document_save_in_progress_set)
    last_xref_was_old_style = property(_mupdf.pdf_document_last_xref_was_old_style_get, _mupdf.pdf_document_last_xref_was_old_style_set)
    has_linearization_object = property(_mupdf.pdf_document_has_linearization_object_get, _mupdf.pdf_document_has_linearization_object_set)
    map_page_count = property(_mupdf.pdf_document_map_page_count_get, _mupdf.pdf_document_map_page_count_set)
    rev_page_map = property(_mupdf.pdf_document_rev_page_map_get, _mupdf.pdf_document_rev_page_map_set)
    fwd_page_map = property(_mupdf.pdf_document_fwd_page_map_get, _mupdf.pdf_document_fwd_page_map_set)
    page_tree_broken = property(_mupdf.pdf_document_page_tree_broken_get, _mupdf.pdf_document_page_tree_broken_set)
    repair_attempted = property(_mupdf.pdf_document_repair_attempted_get, _mupdf.pdf_document_repair_attempted_set)
    repair_in_progress = property(_mupdf.pdf_document_repair_in_progress_get, _mupdf.pdf_document_repair_in_progress_set)
    non_structural_change = property(_mupdf.pdf_document_non_structural_change_get, _mupdf.pdf_document_non_structural_change_set)
    file_reading_linearly = property(_mupdf.pdf_document_file_reading_linearly_get, _mupdf.pdf_document_file_reading_linearly_set)
    file_length = property(_mupdf.pdf_document_file_length_get, _mupdf.pdf_document_file_length_set)
    linear_page_count = property(_mupdf.pdf_document_linear_page_count_get, _mupdf.pdf_document_linear_page_count_set)
    linear_obj = property(_mupdf.pdf_document_linear_obj_get, _mupdf.pdf_document_linear_obj_set)
    linear_page_refs = property(_mupdf.pdf_document_linear_page_refs_get, _mupdf.pdf_document_linear_page_refs_set)
    linear_page1_obj_num = property(_mupdf.pdf_document_linear_page1_obj_num_get, _mupdf.pdf_document_linear_page1_obj_num_set)
    linear_pos = property(_mupdf.pdf_document_linear_pos_get, _mupdf.pdf_document_linear_pos_set)
    linear_page_num = property(_mupdf.pdf_document_linear_page_num_get, _mupdf.pdf_document_linear_page_num_set)
    hint_object_offset = property(_mupdf.pdf_document_hint_object_offset_get, _mupdf.pdf_document_hint_object_offset_set)
    hint_object_length = property(_mupdf.pdf_document_hint_object_length_get, _mupdf.pdf_document_hint_object_length_set)
    hints_loaded = property(_mupdf.pdf_document_hints_loaded_get, _mupdf.pdf_document_hints_loaded_set)
    hint_page = property(_mupdf.pdf_document_hint_page_get, _mupdf.pdf_document_hint_page_set)
    hint_shared_ref = property(_mupdf.pdf_document_hint_shared_ref_get, _mupdf.pdf_document_hint_shared_ref_set)
    hint_shared = property(_mupdf.pdf_document_hint_shared_get, _mupdf.pdf_document_hint_shared_set)
    hint_obj_offsets_max = property(_mupdf.pdf_document_hint_obj_offsets_max_get, _mupdf.pdf_document_hint_obj_offsets_max_set)
    hint_obj_offsets = property(_mupdf.pdf_document_hint_obj_offsets_get, _mupdf.pdf_document_hint_obj_offsets_set)
    resources_localised = property(_mupdf.pdf_document_resources_localised_get, _mupdf.pdf_document_resources_localised_set)
    lexbuf = property(_mupdf.pdf_document_lexbuf_get, _mupdf.pdf_document_lexbuf_set)
    js = property(_mupdf.pdf_document_js_get, _mupdf.pdf_document_js_set)
    recalculate = property(_mupdf.pdf_document_recalculate_get, _mupdf.pdf_document_recalculate_set)
    redacted = property(_mupdf.pdf_document_redacted_get, _mupdf.pdf_document_redacted_set)
    resynth_required = property(_mupdf.pdf_document_resynth_required_get, _mupdf.pdf_document_resynth_required_set)
    event_cb = property(_mupdf.pdf_document_event_cb_get, _mupdf.pdf_document_event_cb_set)
    free_event_data_cb = property(_mupdf.pdf_document_free_event_data_cb_get, _mupdf.pdf_document_free_event_data_cb_set)
    event_cb_data = property(_mupdf.pdf_document_event_cb_data_get, _mupdf.pdf_document_event_cb_data_set)
    num_type3_fonts = property(_mupdf.pdf_document_num_type3_fonts_get, _mupdf.pdf_document_num_type3_fonts_set)
    max_type3_fonts = property(_mupdf.pdf_document_max_type3_fonts_get, _mupdf.pdf_document_max_type3_fonts_set)
    type3_fonts = property(_mupdf.pdf_document_type3_fonts_get, _mupdf.pdf_document_type3_fonts_set)
    orphans_max = property(_mupdf.pdf_document_orphans_max_get, _mupdf.pdf_document_orphans_max_set)
    orphans_count = property(_mupdf.pdf_document_orphans_count_get, _mupdf.pdf_document_orphans_count_set)
    orphans = property(_mupdf.pdf_document_orphans_get, _mupdf.pdf_document_orphans_set)
    xfa = property(_mupdf.pdf_document_xfa_get, _mupdf.pdf_document_xfa_set)
    journal = property(_mupdf.pdf_document_journal_get, _mupdf.pdf_document_journal_set)

    def __init__(self):
        _mupdf.pdf_document_swiginit(self, _mupdf.new_pdf_document())
    __swig_destroy__ = _mupdf.delete_pdf_document