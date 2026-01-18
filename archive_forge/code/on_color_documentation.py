from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.VideoClip import ColorClip
 
    Returns a clip made of the current clip overlaid on a color
    clip of a possibly bigger size. Can serve to flatten transparent
    clips (ideal for previewing clips with masks).
    
    :param size: size of the final clip. By default it will be the
       size of the current clip.
    :param bg_color: the background color of the final clip
    :param pos: the position of the clip in the final clip.
    :param col_opacity: should the added zones be transparent ?
    