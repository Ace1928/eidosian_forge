import json
import re
import warnings
from typing import List, Tuple
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_bilibili_subs_and_info(self, url: str) -> Tuple[str, dict]:
    """
        Retrieve video information and transcript for a given BiliBili URL.
        """
    bvid = BV_PATTERN.search(url)
    try:
        from bilibili_api import sync, video
    except ImportError:
        raise ImportError('requests package not found, please install it with `pip install bilibili-api-python`')
    if bvid:
        v = video.Video(bvid=bvid.group(), credential=self.credential)
    else:
        aid = AV_PATTERN.search(url)
        if aid:
            v = video.Video(aid=int(aid.group()[2:]), credential=self.credential)
        else:
            raise ValueError(f'Unable to find a valid video ID in URL: {url}')
    video_info = sync(v.get_info())
    video_info.update({'url': url})
    if not self.credential:
        return ('', video_info)
    sub = sync(v.get_subtitle(video_info['cid']))
    sub_list = sub.get('subtitles', [])
    if sub_list:
        sub_url = sub_list[0].get('subtitle_url', '')
        if not sub_url.startswith('http'):
            sub_url = 'https:' + sub_url
        response = requests.get(sub_url)
        if response.status_code == 200:
            raw_sub_titles = json.loads(response.content).get('body', [])
            raw_transcript = ' '.join([c['content'] for c in raw_sub_titles])
            raw_transcript_with_meta_info = f'Video Title: {video_info['title']}, description: {video_info['desc']}\n\nTranscript: {raw_transcript}'
            return (raw_transcript_with_meta_info, video_info)
        else:
            warnings.warn(f'Failed to fetch subtitles for {url}. HTTP Status Code: {response.status_code}')
    else:
        warnings.warn(f'No subtitles found for video: {url}. Returning empty transcript.')
    return ('', video_info)